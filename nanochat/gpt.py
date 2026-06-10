"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    """模型配置数据"""
    sequence_len: int = 2048 # 最大序列长度，即模型一次能处理的 Token 数量上限
    vocab_size: int = 32768 # 词表大小，即模型能识别的不同 Token 的总数
    n_layer: int = 12 # Transformer 的层数（解码器层数）
    # 当 n_kv_head == n_head 时，就是标准的 MHA（多头注意力）
    # 如果 n_kv_head < n_head，就是 GQA，多个 Q 头共享一组 KV 头，可以大幅减少显存占用（Llama 2 和 Mistral 就常用这个技巧）
    n_head: int = 6 # query 头数量
    n_kv_head: int = 6 # key/value 头数量
    n_embd: int = 768 # 词嵌入维度，即每个 Token 被映射成的向量长度
    # 滑动窗口注意力模式字符串，在各层上平铺。最终层始终为 L。
    # 字符含义：L=长上下文（完整上下文），S=短上下文（四分之一上下文）
    # 示例："L"=全部完整上下文，"SL"=交替，"SSL"=两个短窗口接一个长窗口
    window_pattern: str = "SSSL"


def norm(x):
    """对 Tensor 的最后一维做均方根归一化操作（RMSNorm）
    """
    return F.rms_norm(x, (x.size(-1),)) # 注意这将在 bf16 下运行，看起来没问题

class Linear(nn.Linear):
    """在 forward 中转换权重以匹配输入数据类型的 nn.Linear。
    替代自动混合精度：主权重保持 fp32 以保证优化器精度，
    但矩阵乘法在激活值的 dtype（通常来自嵌入层的 bf16）下运行。"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """返回 GPT 层是否应该拥有值嵌入（交替模式，最后一层始终包含）。"""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    """旋转位置编码，给注意力层注入位置信息"""
    assert x.ndim == 4  # 多头注意力必须是 4 维张量，确保数据格式为 (B, T, H, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # 将 x 的最后一维切成两半
    # 旋转向量
    # [ y1 ]   [ cos  sin ] [ x1 ]
    # [ y2 ] = [-sin  cos ] [ x2 ]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    """因果自注意力，也就是带有掩码的自注意力，每个 token 只能看到自身及它之前的 token，
    不能看到未来的 token。"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx      # 层的编号，0, 1, 2, ..., n_layer-1
        self.n_head = config.n_head     # query 头数量
        self.n_kv_head = config.n_kv_head   # key/value 头数量
        self.n_embd = config.n_embd     # 词嵌入维度
        self.head_dim = self.n_embd // self.n_head  # query 头的维度
        assert self.n_embd % self.n_head == 0   # 保证词嵌入维度是 query 头数的倍数
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0  # 保证 query 头数量大于等于 key/value 头数量且能整除
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)     # 生成 query 的线性投影层
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # 生成 key 的线性投影层
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)  # 生成 value 的线性投影层
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)  # 输出投影层
        self.ve_gate_channels = 12  # 指定了 value embedding 的通道数
        # value embedding 门控的线性层，只有偶数层存在
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        """
        Args:
            x: 隐藏状态输入，形状 (B, T, C)
            ve: 值嵌入（value embeddings）张量或 None
            cos_sin: 一个包含 (cos, sin) 的元组，用于旋转位置编码
        """
        B, T, C = x.size()  # B = batch size，T = 序列长度，C = n_embd（词嵌入维度）

        # 对输入进行投影以获取 queries, keys 和 values
        # 形状：(B, T, H, D) - FA3 的原生布局，无需转置！
        # B = batch size，T = 序列长度，H = 头数量，D = 头维度
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # 值残差（ResFormer）：使用每个头的输入相关门控混合值嵌入
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # 对 queries 和 keys 应用旋转位置编码以获得相对位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK 归一化
        q = q * 1.2  # 更锐利的注意力（在 Q 和 K 之间分配缩放因子），TODO 思考更优方案
        k = k * 1.2

        # Flash Attention（在 Hopper+ 上使用 FA3，其他环境回退到 PyTorch SDPA）
        # window_size 是 (left, right) 元组：(N, 0) 表示因果注意力，(-1, 0) 表示完整上下文
        if kv_cache is None:
            # 训练：因果注意力，可选择滑动窗口
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # 推理：使用 flash_attn_with_kvcache，它处理缓存管理
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # 在最后一层处理完后推进位置
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # 重新组合注意力头并投影回残差流
        # contiguous 保证了新张量在内存中连续存储
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """标准的 Transformer 前馈神经网络"""
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    """一个包含自注意力和前馈神经网络的模块"""
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: 这个 __init__ 函数在元设备上下文中运行 (!!)
        因此，这里面的任何计算都只涉及形状和数据类型，没有实际数据。
        => 我们实际上是在 init_weights() 中初始化所有数据（参数、缓冲区等）。
        """
        super().__init__()
        self.config = config
        # 为滑动窗口注意力计算每层的窗口大小
        # window_size 是 (left, right) 元组：(-1, 0) 表示完整上下文，(N, 0) 表示滑动窗口
        self.window_sizes = self._compute_window_sizes(config)
        # 为了效率（DDP，张量核心）而对词汇表进行填充，向 pad_vocab_size_to 取整。
        # 这只是一个优化 - 输出会在 forward() 中被裁剪。
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # 每层可学习的标量（灵感来自 modded-nanogpt）
        # resid_lambdas：缩放每层的残差流（初始值 1.0 = 中性）
        # x0_lambdas：将初始嵌入每层混合回来（初始值 0.0 = 禁用）
        # 分开参数，以便它们可以有不同的优化器处理方式
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # 假初始化，真正初始在 init_weights() 中
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # 假初始化，真正初始在 init_weights() 中
        # Smear：将前一个 token 的嵌入混合到当前 token 中（廉价的类 bigram 信息）
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout：在最终归一化前减去缓存的中间层残差，以移除低层特征
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value 嵌入（ResFormer 风格）：交替层，最后一层始终包含
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # 为了支持元设备初始化，我们在这里初始化旋转位置编码，但只包含“假”的元张量。
        # 至于 rotary_seq_len，这些旋转位置编码在内存中非常小，
        # 所以我们只需将其过度计算 10 倍，但如果真的达到那个数量，会断言失败。
        # 将来我们可以动态地增长缓存，目前来说这没问题。
        self.rotary_seq_len = config.sequence_len * 10 # 10 倍过度计算应该足够了，TODO 以后可以做得更好
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False 意味着它不会保存到检查点中
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        在这个函数中初始化整个模型，以求最大清晰度。

        wte（嵌入）：     正态分布，std=1.0
        lm_head：         正态分布，std=0.001
        对于每个 block：
            attn.c_q：        均匀分布，std=1/sqrt(n_embd)
            attn.c_k：        均匀分布，std=1/sqrt(n_embd)
            attn.c_v：        均匀分布，std=1/sqrt(n_embd)
            attn.c_proj：     全零
            mlp.c_fc：        均匀分布，std=1/sqrt(n_embd)
            mlp.c_proj：      全零
        """

        # 嵌入和逆嵌入，使用正态分布初始化参数
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer 块：均匀初始化
        # sqrt((b - a)^2 / 12) = 1 / sqrt(n_embd), b = -a = s
        # s = sqrt(3 / n_embd)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # 权重使用 Uniform 以避免离群值
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # 投影层初始化为零
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # c_fc 的初始化缩放为 0.4 倍
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # 每层标量
        # 每层 resid 初始化：浅层残差强度大，深层残差强度小
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # 衰减的 x0 初始化：浅层获得更多输入嵌入混合
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # 值嵌入（像 c_v 一样初始化：相同 std 的均匀分布）
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # 门控权重初始化为小的正值，使门控从略高于中性开始
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # 旋转位置编码
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # 将嵌入转换为 COMPUTE_DTYPE：优化器可以容忍降低精度的嵌入，且能节省内存。
        # 例外：fp16 需要 fp32 嵌入，因为 GradScaler 无法反缩放 fp16 梯度。
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        """旋转位置编码"""
        # TODO: 提高 base theta？例如 100K 是最近更常见的值
        # 从模型嵌入中自动检测设备
        if device is None:
            device = self.transformer.wte.weight.device
        # 步进通道
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 步进时间步
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算每个（时间，通道）对上的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # 添加 batch 和 head 维度以便后续广播
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        计算滑动窗口注意力中每层的窗口大小，目前所有的窗口 right = 0，left = 长上下文或短上下文。

        返回 FA3 的 window_size 参数的 (left, right) 元组列表：
        - left：当前位置之前要关注多少个 token（-1 表示无限制）
        - right：当前位置之后要关注多少个 token（因果注意力为 0）

        模式字符串在各层之间平铺。最后一层始终使用 L（完整上下文）。
        字符含义：L=长上下文（完整上下文），S=短上下文（四分之一上下文）

        例如 SSL，n_layer = 6，结果为:
        (short_window, 0)(short_window, 0)(long_window, 0)
        (short_window, 0)(short_window, 0)(long_window, 0)
        """
        # 检查格式
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # 将字符映射到窗口大小
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # 向上取整到 FA3 的 tile 大小（2048 -> 768）
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # 跨层平铺模式，例如 12 层，模式为 SSSL，结果为 SSSLSSSLSSSL
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 最后一层始终使用完整上下文
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        估算模型每个 token 的 FLOPs（前向 + 反向）。
        每个 matmul 权重参数在前向中贡献 2 次 FLOPs（乘法 + 累加），反向中贡献前向的 2 倍 => 2+4=6。
        最清晰的解释：https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        除此之外，12 * h * q * effective_seq_len 用于注意力机制中的 key @ query 矩阵乘法 FLOPs。
        对于滑动窗口，effective_seq_len 每层不同（受窗口大小限制）。
        参考：https://arxiv.org/abs/2204.02311 (PaLM 论文)。
        这个估算与 Chinchilla 论文中的精确公式约有 1% 的偏差，差异在于：
        - Chinchilla 将嵌入层也算作 FLOPs（？奇怪，它只是一个查表操作 => 我们忽略）
        - Chinchilla 将注意力 softmax 中的 exp/求和/除法算作 FLOPs（有点可疑且非常微小 => 我们忽略）
        """
        nparams = sum(p.numel() for p in self.parameters())
        # 排除非 matmul 参数：嵌入和每层标量
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # 计算每层注意力 FLOPs 的总和，考虑滑动窗口
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) 元组，我们使用 left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()   # B = batch size，T = 序列长度

        # 获取当前序列长度的旋转位置编码（形状为 (1, seq_len, 1, head_dim/2)）
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # 如果存在 kv 缓存，需要将旋转位置编码偏移到缓存的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # 将缓存截断到当前序列长度

        # 嵌入 token
        x = self.transformer.wte(idx) # 嵌入当前 token
        x = x.to(COMPUTE_DTYPE) # 确保激活值在计算 dtype 中（通常无操作，但对 fp16 代码路径有效）
        x = norm(x)

        # Smear：将前一个 token 的嵌入混合到当前位置（廉价的 bigram 信息）
        if kv_cache is None:
            # 训练 / 朴素生成：完整序列可用，使用快速切片
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV 缓存推理：从缓存中读取上一个嵌入，存储当前嵌入用于下一步
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # 预填充：对位置 1+ 应用 smear，与训练相同
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # 解码：单个 token，使用缓存的上一个嵌入
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # 前向传播 Transformer 主干
        x0 = x  # 保存归一化后的初始嵌入，用于 x0 残差
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # 在中点处缓存
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # 在 logit 投影之前减去中间层残差，以移除低层特征
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # 前向传播 lm_head（计算 logits）
        softcap = 15 # 将 logits 平滑限制到范围 [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- 非常大的张量，占用大量内存
        logits = logits[..., :self.config.vocab_size] # 切片去掉填充部分
        logits = logits.float() # 切换到 fp32 用于 logit softcap 和损失计算
        logits = softcap * torch.tanh(logits / softcap) # 压缩 logits

        if targets is not None:
            # 训练：给定 targets，计算并返回损失
            # TODO 实验分块交叉熵？
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # 推理：直接返回 logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
