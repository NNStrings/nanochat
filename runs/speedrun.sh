#!/bin/bash

# 此脚本用于训练你自己的 GPT-2 级别的 LLM（预训练 + 微调）
# 它设计为在全新的 8XH100 GPU 节点上运行，大约需要 3 小时完成。

# 1) 示例启动（最简单）：
# bash runs/speedrun.sh
# 2) 在 screen 会话中启动示例（因为运行需要约 3 小时）：
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) 启用 wandb 日志记录的启动示例，但请先参考下文设置 wandb：
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# 默认的中间产物目录位于 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 使用 uv 进行 Python venv 环境设置

# 安装 uv（如果尚未安装）
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 创建本地虚拟环境 .venv（如果不存在）
[ -d ".venv" ] || uv venv
# 安装本仓库依赖
uv sync --extra gpu
# 激活 venv，使得 `python` 使用项目虚拟环境而不是系统 Python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb 设置
# 如果你想使用 wandb 进行日志记录（很好用，推荐）。
# 1) 首先确保登录 wandb，例如运行：
#    `wandb login`
# 2) 在运行此脚本时设置 WANDB_RUN 环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"：它作为特殊情况处理，跳过向 wandb 记录日志
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 在运行过程中，我们会将 markdown 报告写入 base 目录下的 report/ 目录。
# 此命令会清空该目录并写入一个头部部分，其中包含大量系统信息和一个标记运行开始的时间戳。
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# 下载预训练数据集的前约 20 亿字符
# 每个数据分片约 2.5 亿字符
# 因此此时我们下载 2e9 / 250e6 = 8 个数据分片
# 每个分片压缩后约为 100MB 的文本，因此磁盘上大约占用 800MB 数据
# 关于此数据如何准备的详细信息，请参阅 dev/repackage_data_reference.py
python -m nanochat.dataset -n 8
# 立即在后台启动下载更多分片，同时 tokenizer 进行训练
# 达到 GPT-2 能力预训练大约需要 150 个分片，额外增加 20 个作为填充。
# 整个数据集中可用的最大分片总数为 6542。
# 节省资源，暂时不需要
# python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# 在约 20 亿字符数据上训练 tokenizer，词汇表大小为 2**15 = 32768
python -m scripts.tok_train
# 评估 tokenizer（报告压缩比等）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 基座模型（预训练）
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# # d24 模型（略微欠训练以击败 GPT-2 => 将数据:参数比从计算最优值 10.5（默认）降低到 8）
# torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# # 评估模型：CORE 指标，训练/验证集上的 BPB，并生成样本
# torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# d24 模型（略微欠训练以击败 GPT-2 => 将数据:参数比从计算最优值 10.5（默认）降低到 8）
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=4 --target-param-data-ratio=8 --device-batch-size=1 --fp8 --run=$WANDB_RUN
# 评估模型：CORE 指标，训练/验证集上的 BPB，并生成样本
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --device-batch-size=1

# -----------------------------------------------------------------------------
# SFT（教会模型对话特殊 token、工具使用、多项选择）

# 下载 2.3MB 的合成身份对话，为 nanochat 赋予个性
# 关于此数据如何准备以及如何轻松调整它的说明，请参阅 dev/gen_synthetic_data.py
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# # 运行 SFT 并评估模型
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
# torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- --device-batch-size=1 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft

# 通过 CLI 与模型对话！去掉 -p 参数可以以交互方式聊天
# python -m scripts.chat_cli -p "Why is the sky blue?"

# 更好的是，通过漂亮的 WebUI ChatGPT 风格与模型对话
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 通过汇总所有部分生成完整报告
# report.md 是输出文件，并将被复制到当前目录以便使用
python -m nanochat.report generate
