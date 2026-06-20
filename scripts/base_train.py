"""
训练模型。从项目的根目录运行，命令如下：

python -m scripts.base_train

或分布式运行：

torchrun --nproc_per_node=8 -m scripts.base_train

如果只在 CPU/Macbook 上运行，你可能需要训练一个非常小的 LLM。示例：
python -m scripts.base_train \
    --depth=4 --max-seq-len=512 \
    --device-batch-size=1 \
    --eval-tokens=512 \
    --core-metric-every=-1 \
    --total-batch-size=512 \
    --num-iterations=20
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
import math
import argparse
from dataclasses import asdict
from contextlib import contextmanager

import wandb
import torch
import torch.distributed as dist

from nanochat.gpt import GPT, GPTConfig, Linear
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit, tokenizing_distributed_data_loader_with_state_bos_bestfit
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type, get_peak_flops, COMPUTE_DTYPE, COMPUTE_DTYPE_REASON, is_ddp_initialized
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from nanochat.flash_attention import HAS_FA3
from scripts.base_eval import evaluate_core
print_banner()

# -----------------------------------------------------------------------------
# CLI 参数
parser = argparse.ArgumentParser(description="预训练基础模型")
# 日志记录
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' 禁用 wandb 日志记录）")
# 运行时
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps（为空则自动检测）")
# FP8 训练
parser.add_argument("--fp8", action="store_true", help="启用 FP8 训练（需要 H100+ GPU 和 torchao）")
parser.add_argument("--fp8-recipe", type=str, default="tensorwise", choices=["rowwise", "tensorwise"], help="FP8 缩放策略：tensorwise（更快，推荐）或 rowwise（更准确但更慢）")
# 模型架构
parser.add_argument("--depth", type=int, default=20, help="Transformer 模型的深度")
parser.add_argument("--aspect-ratio", type=int, default=64, help="model_dim = depth * aspect_ratio")
parser.add_argument("--head-dim", type=int, default=128, help="注意力机制的目标头维度")
parser.add_argument("--max-seq-len", type=int, default=2048, help="最大上下文长度")
parser.add_argument("--window-pattern", type=str, default="SSSL", help="滑动窗口模式，在各层上平铺：L=全局注意力，S=半上下文窗口（例如 'SSL'）")
# 训练范围（仅使用一个，按优先级顺序）
parser.add_argument("--num-iterations", type=int, default=-1, help="显式指定优化步数（-1 = 禁用）")
parser.add_argument("--target-flops", type=float, default=-1.0, help="计算达到 target_flops 所需的 num_iterations（-1 = 禁用）")
parser.add_argument("--target-param-data-ratio", type=float, default=12, help="计算维持 data:param 比例所需的 num_iterations（Chinchilla=20，-1 = 禁用）")
# 优化
parser.add_argument("--device-batch-size", type=int, default=32, help="每设备的 batch 大小。如果显存不足（OOM），可降为 16、8、4...")
parser.add_argument("--total-batch-size", type=int, default=-1, help="总 batch 大小（以 token 计）。合适的数值例如 524288。（-1 = 自动计算最优值）")
parser.add_argument("--embedding-lr", type=float, default=0.3, help="embedding 参数的学习率（Adam）")
parser.add_argument("--unembedding-lr", type=float, default=0.008, help="unembedding 参数的学习率（Adam）")
parser.add_argument("--weight-decay", type=float, default=0.28, help="Muon 优化器（用于权重）的谨慎权重衰减")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数的学习率（Muon）")
parser.add_argument("--scalar-lr", type=float, default=0.5, help="标量参数（resid_lambdas, x0_lambdas）的学习率")
parser.add_argument("--warmup-steps", type=int, default=40, help="学习率预热的步数")
parser.add_argument("--warmdown-ratio", type=float, default=0.65, help="学习率衰减阶段占迭代次数的比例")
parser.add_argument("--final-lr-frac", type=float, default=0.05, help="最终学习率占初始学习率的比例")
parser.add_argument("--resume-from-step", type=int, default=-1, help="从指定步数恢复训练（-1 = 禁用）")
# 评估
parser.add_argument("--eval-every", type=int, default=250, help="每 N 步评估验证集的 bpb（-1 = 禁用）")
parser.add_argument("--eval-tokens", type=int, default=80*524288, help="用于评估验证集损失的 token 数量")
parser.add_argument("--core-metric-every", type=int, default=2000, help="每 N 步评估 CORE 指标（-1 = 禁用）")
parser.add_argument("--core-metric-max-per-task", type=int, default=500, help="CORE 指标中每个任务的最大样本数")
parser.add_argument("--sample-every", type=int, default=2000, help="每 N 步从模型中进行采样（-1 = 禁用）")
parser.add_argument("--save-every", type=int, default=-1, help="每 N 步保存检查点（-1 = 仅在结束时保存）")
# 输出
parser.add_argument("--model-tag", type=str, default=None, help="覆盖用于检查点目录名称的 model tag")
args = parser.parse_args()
# 日志记录参数信息
user_config = vars(args).copy()
# -----------------------------------------------------------------------------
# 计算初始化和 wandb 日志

# 设备类型
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
# 分布式初始化与设备对象
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
# 只有 rank 0 的进程（主进程）负责输出日志、保存 checkpoint、与 wandb 交互等，避免多进程重复操作
master_process = ddp_rank == 0
# 同步函数与显存监控（CUDA 专用）
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0
# GPU 性能信息获取
if device_type == "cuda":
    gpu_device_name = torch.cuda.get_device_name(0)
    gpu_peak_flops = get_peak_flops(gpu_device_name)
    print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
else:
    gpu_peak_flops = float('inf')  # MFU not meaningful for CPU/MPS
print0(f"COMPUTE_DTYPE: {COMPUTE_DTYPE} ({COMPUTE_DTYPE_REASON})")

# wandb 日志初始化
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=args.run, config=user_config)

# 获取 Flash Attention 状态，检查 Flash Attention 3 是否可用
from nanochat.flash_attention import USE_FA3
using_fa3 = USE_FA3
if using_fa3:
    print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
else:
    print0("!" * 80)
    if HAS_FA3 and COMPUTE_DTYPE != torch.bfloat16:
        print0(f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={COMPUTE_DTYPE}. Using PyTorch SDPA fallback")
    else:
        print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
    print0("WARNING: Training will be less efficient without FA3")
    if args.window_pattern != "L":
        print0(f"WARNING: SDPA has no support for sliding window attention (window_pattern='{args.window_pattern}'). Your GPU utilization will be terrible.")
        print0("WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns.")
    print0("!" * 80)

# -----------------------------------------------------------------------------
# 分词器对评估很有用，而且我们需要词汇表大小来初始化模型
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# -----------------------------------------------------------------------------
# 初始化模型

def build_model_meta(depth):
    """在 meta 设备上为给定的深度构建模型（仅形状/数据类型，无数据）"""
    # 模型维度向上取整到 head_dim 的最近倍数，以便整齐划分
    # （FA3 要求 head_dim 能被 8 整除，这保证了 head_dim == args.head_dim 精确匹配）
    base_dim = depth * args.aspect_ratio
    model_dim = ((base_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = model_dim // args.head_dim
    config = GPTConfig(
        sequence_len=args.max_seq_len, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=args.window_pattern,
    )
    # 在 meta 上下文中创建：不分配任何内存
    # 只保留元数据：张量对象仅记录形状、数据类型、设备类型等必要信息，数据缓冲区为 None
    # 可以正常进行模型前向的形状计算，验证模型结构是否正确
    with torch.device("meta"):
        model_meta = GPT(config)
    return model_meta

# 构建模型，移动到设备上，初始化权重
model = build_model_meta(args.depth) # 1) 在元设备上构建（仅形状/数据类型，无数据）
model_config = model.config
model_config_kwargs = asdict(model_config)
print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
model.to_empty(device=device) # 2) 所有张量在目标设备上获得存储空间，但包含未初始化（垃圾）数据
model.init_weights() # 3) 所有张量被初始化

# 如果要恢复运行，则使用检查点的参数覆盖模型参数
base_dir = get_base_dir()
output_dirname = args.model_tag if args.model_tag else f"d{args.depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = args.resume_from_step != -1
if resuming:
    print0(f"Resuming optimization from step {args.resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, args.resume_from_step, device, load_optimizer=True, rank=ddp_rank)
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # 复制完成后释放这部分内存

# -----------------------------------------------------------------------------
# FP8 训练初始化和管理（这必须在 torch.compile 之前完成）

# 如果设置了 --fp8，将 Linear 层转换为 Float8Linear
if args.fp8:
    if device_type != "cuda":
        print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
    else:
        # 我们自定义的 fp8 比 torchao 更简单，为精确的 API 兼容性而编写
        from nanochat.fp8 import Float8LinearConfig, convert_to_float8_training
        # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        import torch.nn as nn

        # 过滤条件：维度必须能被 16 整除（FP8 硬件要求）且足够大
        def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
            if not isinstance(mod, nn.Linear):
                return False
            if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                return False
            if min(mod.in_features, mod.out_features) < 128:
                return False
            return True

        fp8_config = Float8LinearConfig.from_recipe_name(args.fp8_recipe)
        num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
        convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
        num_fp8 = sum(1 for m in model.modules() if 'Float8' in type(m).__name__)
        num_skipped = num_linear - num_fp8
        print0(f"✓ FP8 training enabled ({args.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)")

# Context manager to temporarily disable FP8 so that model evaluation remains in BF16
@contextmanager
def disable_fp8(model):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    import torch.nn as nn

    # Find all Float8Linear modules and their locations
    fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
    for name, module in model.named_modules():
        if 'Float8' in type(module).__name__:
            if '.' in name:
                parent_name, attr_name = name.rsplit('.', 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield  # No FP8 modules, nothing to do
        return

    # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
    # Use device="meta" to avoid VRAM spike - the weight tensor will be swapped in afterwards
    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device="meta",  # Use meta device to avoid unnecessary VRAM allocation
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight  # share, don't copy
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        # Restore Float8Linear modules
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)

# -----------------------------------------------------------------------------
# 编译模型

orig_model = model # 原始未编译的模型，用于保存原始模型 state_dict 以及推理/评估（因为形状可能会变化）
# 将模型编译为更高效的中间表示，从而加速模型的前向传播和反向传播
model = torch.compile(model, dynamic=False) # 模型的输入形状永远不会改变，因此 dynamic=False 是安全的

# -----------------------------------------------------------------------------
# 缩放法则和 muP 外推，用于确定最优训练范围、批次大小、学习率、权重衰减

# 获取我们模型的参数数量
param_counts = model.num_scaling_params()
print0(f"Parameter counts:")
for key, value in param_counts.items():
    print0(f"{key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 1) 使用缩放法则确定最优的训练范围（以 token 数计）
# 计算最优模型满足 --target-param-data-ratio 所定义的 Token:参数 比例（通过缩放法则分析实验得出）
# 我们已经初始化了模型，因此已知参数数量。最优 token 数现在就是 target-param-data-ratio * 参数数量
def get_scaling_params(m):
    # 至于具体使用哪些参数，transformer 矩阵 + lm_head 给出最清晰的缩放法则（参见 dev/LOG.md 2026年1月27日）
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
num_scaling_params = get_scaling_params(model)
target_tokens = int(args.target_param_data_ratio * num_scaling_params) # 即将训练的模型的最优 token 数

# 我们的参考模型是 d12，大量超参数在此调优，然后迁移到更高深度（muP 风格）
d12_ref = build_model_meta(12) # 在元设备上创建模型
D_REF = args.target_param_data_ratio * get_scaling_params(d12_ref) # d12 的计算最优训练范围（以 token 数计，经验测量得出）
B_REF = 2**19 # d12 的最优批次大小约等于 524,288 个 token（经验测量得出）

# 2) 现在有了 token 训练范围，我们可以计算最优批次大小
# 遵循 Power Lines 论文（Bopt ∝ D^0.383），参考文献：https://arxiv.org/abs/2505.13738
# 最优批次大小大约按 D^0.383 增长，例如如果 D 从 d12 翻倍到 d24，B 应增长约 2^0.383 ≈ 1.3 倍。
total_batch_size = args.total_batch_size # 用户提供的覆盖值是可能的
if total_batch_size == -1:
    batch_size_ratio = target_tokens / D_REF
    predicted_batch_size = B_REF * batch_size_ratio ** 0.383
    total_batch_size = 2 ** round(math.log2(predicted_batch_size)) # 为效率取整到最近的 2 的幂
    print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")

# 3) 知道批次大小后，我们现在可以计算学习率修正（更大的批次大小允许更高的学习率）
batch_lr_scale = 1.0
batch_ratio = total_batch_size / B_REF # B/B_ref
if batch_ratio != 1.0:
    # SGD：批次大小的线性缩放是标准做法（nanochat 中未使用）
    # AdamW：平方根缩放是标准做法：η ∝ √(B/B_ref)
    # Muon：我们将对 Muon 使用与 AdamW 相同的缩放：η ∝ √(B/B_ref)（未仔细研究，假设！）
    batch_lr_scale = batch_ratio ** 0.5 # η ∝ √(B/B_ref)
    print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")

# 4) 知道批次大小和 token 训练范围后，我们现在可以计算合适的权重衰减缩放
# 我们采用来自 https://arxiv.org/abs/2405.13698 的 T_epoch 框架
# 论文的核心思想是 T_epoch = B/(η·λ·D) 应保持恒定。
# 上面我们使用了学习率缩放 η ∝ √(B/B_ref)。因此通过大约 10 行数学推导，要保持 T_epoch 恒定，我们需要：
# λ = λ_ref · √(B/B_ref) · (D_ref/D)
# 注意这些论文研究的是 AdamW，*不是* Muon。我们盲目遵循 AdamW 的缩放理论，希望它对 Muon 也大致有效。
weight_decay_scaled = args.weight_decay * math.sqrt(total_batch_size / B_REF) * (D_REF / target_tokens)
if weight_decay_scaled != args.weight_decay:
    print0(f"Scaling weight decay from {args.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {args.depth}")

# -----------------------------------------------------------------------------
# 初始化优化器（组合版 MuonAdamW：矩阵参数使用 Muon，其余参数使用 AdamW）
optimizer = model.setup_optimizer(
    # AdamW 超参数
    unembedding_lr=args.unembedding_lr * batch_lr_scale,
    embedding_lr=args.embedding_lr * batch_lr_scale,
    scalar_lr=args.scalar_lr * batch_lr_scale,
    # Muon 超参数
    matrix_lr=args.matrix_lr * batch_lr_scale,
    weight_decay=weight_decay_scaled,
)

# 从检查点恢复
if resuming:
    optimizer.load_state_dict(optimizer_data)
    del optimizer_data

# -----------------------------------------------------------------------------
# GradScaler 用于 fp16 训练（bf16/fp32 不需要 — bf16 与 fp32 具有相同的指数范围）
scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
if scaler is not None:
    print0("GradScaler enabled for fp16 training")

# -----------------------------------------------------------------------------
# 初始化训练/验证的数据加载器
dataloader_resume_state_dict = None if not resuming else meta_data["dataloader_state_dict"]
train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(tokenizer, args.device_batch_size, args.max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # 启动第一个数据批次的加载

# -----------------------------------------------------------------------------
# 计算我们将训练的迭代次数并设置各种调度器

# num_iterations：要么由用户指定，要么来自目标 FLOPs，要么来自目标数据：参数比例（按此顺序）
assert args.num_iterations > 0 or args.target_param_data_ratio > 0 or args.target_flops > 0
if args.num_iterations > 0:
    # 如果提供了 num_iterations，则覆盖为特定值
    num_iterations = args.num_iterations
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif args.target_flops > 0:
    # 根据目标 FLOPs 计算迭代次数（用于缩放法则分析，例如 runs/scaling_laws.sh）
    num_iterations = round(args.target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif args.target_param_data_ratio > 0:
    # 根据目标参数数据比例计算迭代次数（最常见的使用场景）
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations # 我们将实际训练的 token 总数
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}") # 例如 Chinchilla 约为 20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# 学习率调度（线性预热，恒定，线性衰减）
def get_lr_multiplier(it):
    """
    根据迭代步数线性调整学习率乘子，1/warmup_iters -> 1 -> args.final_lr_frac

    Args:
        it:
            训练步数
    """
    warmup_iters = args.warmup_steps
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * args.final_lr_frac

# Muon 优化器的动量调度（预热到 0.97，在学习率衰减期间衰减到 0.90）
def get_muon_momentum(it):
    warmdown_iters = round(args.warmdown_ratio * num_iterations)
    warmdown_start = num_iterations - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / warmdown_iters
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97

# Muon 优化器的权重衰减调度（在训练过程中余弦衰减到零）
def get_weight_decay(it):
    return weight_decay_scaled * 0.5 * (1 + math.cos(math.pi * it / num_iterations))

# -----------------------------------------------------------------------------
# 训练循环

# 循环状态（由训练循环更新的变量）
if not resuming:
    step = 0
    val_bpb = None # 如果 eval_every > 0 则会被设置
    min_val_bpb = float("inf")
    smooth_train_loss = 0 # 训练损失的 EMA（指数移动平均）
    total_training_time = 0 # 训练的总挂钟时间
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    val_bpb = meta_data["val_bpb"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_train_loss = loop_state["smooth_train_loss"]
    total_training_time = loop_state["total_training_time"]

# 计算达到每步所需总批次大小所需的梯度累积微步数
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # 单个 rank 每次迭代的 token 数
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # 所有 rank 每次迭代的总 token 数
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# 开始！
while True:
    last_step = step == num_iterations # 循环运行 num_iterations+1 次，以便在结束时进行评估/保存
    flops_so_far = num_flops_per_token * total_batch_size * step

    # 每eval_every步评估验证集的 bpb（所有 rank 参与）
    if args.eval_every > 0 and (last_step or step % args.eval_every == 0):
        model.eval()    # 调节 training 值，在验证中使用，和后面的 model.train() 对应
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with disable_fp8(model):
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        model.train()

    # 偶尔评估 CORE 指标（所有 rank 参与）
    # 使用原始的未编译模型，因为输入形状在不断变化
    # 在评估时禁用 FP8，以使用 BF16 获得更一致/准确的结果
    results = {}
    if args.core_metric_every > 0 and (last_step or (step > 0 and step % args.core_metric_every == 0)):
        model.eval()
        with disable_fp8(orig_model):
            results = evaluate_core(orig_model, tokenizer, device, max_per_task=args.core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        })
        model.train()

    # 偶尔从模型中采样（仅在主进程上执行）
    # 使用原始的未编译模型，因为输入形状在不断变化
    if args.sample_every > 0 and master_process and (last_step or (step > 0 and step % args.sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # 使用 orig_model 以避免重新编译
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with disable_fp8(orig_model):
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != args.resume_from_step and args.save_every > 0 and step % args.save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # 模型参数
            optimizer.state_dict(), # 优化器状态
            { # 以 json 格式保存的元数据
                "step": step, # 迭代步数
                "val_bpb": val_bpb, # 最后一步的损失
                "model_config": model_config_kwargs,
                "user_config": user_config, # 用户训练脚本输入的参数
                "device_batch_size": args.device_batch_size,
                "max_seq_len": args.max_seq_len,
                "total_batch_size": total_batch_size,
                "dataloader_state_dict": dataloader_state_dict, # 数据加载器的状态
                "loop_state": { # 所有循环状态（除 step 外），以便恢复训练
                    "min_val_bpb": min_val_bpb,
                    "smooth_train_loss": smooth_train_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # 终止条件（TODO：可能还需添加损失爆炸等）
    if last_step:
        break

    # -------------------------------------------------------------------------
    # 单步训练
    # 计算梯度
    synchronize()
    t0 = time.time()
    for micro_step in range(grad_accum_steps):
        loss = model(x, y)  # 调用 forward 计算损失
        train_loss = loss.detach() # 用于日志记录
        loss = loss / grad_accum_steps # 每次 .backward() 是梯度求和 => 此处归一化损失
        if scaler is not None:
            scaler.scale(loss).backward()   # 如果启用了 AMP
        else:
            loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # 在 GPU 忙于前向/反向时预取下一批数据
    # 优化器参数动态调整
    lrm = get_lr_multiplier(step)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    if scaler is not None:
        scaler.unscale_(optimizer)
        # 在分布式训练中，所有 rank 必须就是否跳过该步骤达成一致。
        # 每个 rank 可能独立遇到 inf/nan 梯度，因此我们对 found_inf 标志进行 all-reduce
        # （取最大值 = 如果任何 rank 发现 inf，所有 rank 都跳过）。
        if is_ddp_initialized():
            for v in scaler._found_inf_per_device(optimizer).values():
                dist.all_reduce(v, op=dist.ReduceOp.MAX)
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    model.zero_grad(set_to_none=True)
    train_loss_f = train_loss.item() # .item() 是 CPU-GPU 同步点
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # 日志 (只有 CPU 动作)
    ema_beta = 0.9 # EMA 衰减因子，用于平滑处理，仅为了更美观的日志记录
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f # 对训练损失进行 EMA 平滑
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # 对 EMA 进行去偏校正
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
    if step > 10:
        total_training_time += dt # 仅统计前 10 步之后的时间
    # 根据平均每步时间计算预计剩余时间（排除前 10 步）
    steps_done = step - 10
    if steps_done > 0:
        avg_time_per_step = total_training_time / steps_done
        remaining_steps = num_iterations - step
        eta_seconds = remaining_steps * avg_time_per_step
        eta_str = f" | eta: {eta_seconds/60:.1f}m"
    else:
        eta_str = ""
    epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time/60:.2f}m{eta_str}")
    if step % 100 == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": epoch,
        }
        wandb_run.log(log_data)

    # 状态更新
    first_step_of_run = (step == 0) or (resuming and step == args.resume_from_step)
    step += 1

    # 垃圾回收器有点过于活跃，并且由于一些不太清楚的原因，
    # 它会相当频繁地花费约500ms扫描循环引用，结果每次只清理很少的几个小对象。
    # 所以我们在这里手动管理并帮助它
    if first_step_of_run:
        gc.collect() # 手动收集初始化阶段产生的大量垃圾
        gc.freeze() # 立即冻结当前所有存活对象，并将它们排除在GC之外
        gc.disable() # 核弹级干预：完全禁用GC，但以下情况除外：
    elif step % 5000 == 0: # 每5000步...
        gc.collect() # 手动收集，只是为了在非常非常长的运行中保持安全

# 打印一些状态
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
if val_bpb is not None:
    print0(f"Minimum validation bpb: {min_val_bpb:.6f}")

# 记录报告
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # 用户参数
    { # 训练启动的参数
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
        "DDP world size": ddp_world_size,
        "warmup_steps": args.warmup_steps,
        "warmdown_ratio": args.warmdown_ratio,
        "final_lr_frac": args.final_lr_frac,
    },
    { # 训练结果状态
        "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# 清理
wandb_run.finish() # wandb 运行结束
compute_cleanup()
