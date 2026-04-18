"""
基础/预训练数据集是一组 parquet 文件。
本文件包含以下实用功能：
- 遍历 parquet 文件并从中产出文档
- 如果文件不在磁盘上，则按需下载

有关数据集准备方式的详细信息，请参见 `repackage_data_reference.py`。
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# 当前预训练数据集的具体细节

# 互联网上托管数据，并可按需下载该数据的 URL。
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # 最后一个数据分片是 shard_06542.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # 文件名的格式 shard_00001.parquet
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_climbmix")

# -----------------------------------------------------------------------------
# 这些函数对其他模块是有用的工具，需要被引入

def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """检查数据目录，并返回所有 Parquet 文件的完整路径。"""
    data_dir = DATA_DIR if data_dir is None else data_dir

    # 为兼容从 FinewebEdu-100B 升级至 ClimbMix-400B 而保留的旧版兼容代码
    # 此代码最终将被移除。
    if not os.path.exists(data_dir):
        if warn_on_legacy:
            print()
            print("=" * 80)
            print("  WARNING: DATASET UPGRADE REQUIRED")
            print("=" * 80)
            print()
            print(f"  Could not find: {data_dir}")
            print()
            print("  nanochat recently switched from FinewebEdu-100B to ClimbMix-400B.")
            print("  Everyone who does `git pull` as of March 4, 2026 is expected to see this message.")
            print("  To upgrade to the new ClimbMix-400B dataset, run these two commands:")
            print()
            print("    python -m nanochat.dataset -n 170     # download ~170 shards, enough for GPT-2, adjust as desired")
            print("    python -m scripts.tok_train           # re-train tokenizer on new ClimbMix data")
            print()
            print("  For now, falling back to your old FinewebEdu-100B dataset...")
            print("=" * 80)
            print()

        data_dir = os.path.join(base_dir, "base_data")

    # 获取 data_dir 下所有 .parquet 文件并排序
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    # 补充成完整路径
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    以 parquet 底层的 row_group 为批次迭代数据集，以提高效率。
    - split 参数可取值为 "train" 或 "val"；其中，最后一个 Parquet 文件将被视为验证集（val）。
    - start 和 step 参数有助于在 DDP（分布式数据并行）模式下跳过行数据。例如：start=rank, step=world_size。
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    # 前面的用作训练集，后面的用作验证集
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        # pyarrow 用于高效读写 parquet、处理列式数据
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            # 每次只迭代一个 row_group，将数据中 text 列表转换成 python list 返回
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # 将 index 构造成 shard_{index:05d}.parquet 格式的文件路径
    # 如果文件存在则直接跳过
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # 构造文件的远程 url
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # 带有重试机制的下载
    max_attempts = 5    # 最大重试次数
    for attempt in range(1, max_attempts + 1):
        try:
            # 流式下载，适合大文件，节省内存
            response = requests.get(url, stream=True, timeout=30)
            # 检查  HTTP 状态码，出错会抛出 requests.HTTPError 异常
            response.raise_for_status()
            # 先写入临时文件
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:    # 二进制打开
                # 迭代响应内容，每次读取 1MB（1024*1024 字节）的块
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:       # 过滤掉空的块
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # 清理所有部分文件
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # 尝试几次，采用指数退避策略：2^attempt 秒
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretraining dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of train shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    # 创建 output 目录，DATA_DIR 默认为 `~/.cache/nanochat/base_data_climbmix`
    os.makedirs(DATA_DIR, exist_ok=True)

    # 其工作原理是：用户通过 `-n` 标志指定要下载的训练分片数量。 
    # 此外，验证分片 *总是* 会被下载，并被固定为最后一个分片。
    num_train_shards = MAX_SHARD if args.num_files == -1 else min(args.num_files, MAX_SHARD)
    ids_to_download = list(range(num_train_shards))
    ids_to_download.append(MAX_SHARD)

    # 下载
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    # 创建进程池（默认 4 进程）按照 ids_to_download 列表调用 download_single_file 下载
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # 记录结果
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
