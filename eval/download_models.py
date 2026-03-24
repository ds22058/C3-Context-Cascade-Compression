"""
下载 C3 模型和 Qwen2.5-3B 到本地 ./models/ 目录。
支持通过 HF_ENDPOINT 环境变量设置镜像（如 https://hf-mirror.com）。

用法：
    python eval/download_models.py
    python eval/download_models.py --models c3          # 只下载 C3
    python eval/download_models.py --models qwen        # 只下载 Qwen2.5-3B
    HF_ENDPOINT=https://hf-mirror.com python eval/download_models.py
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


def download(repo_id: str, local_dir: str, desc: str) -> str:
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[跳过] {desc} 已存在: {local_dir}")
        return local_dir

    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"[下载] {desc}")
    print(f"  仓库: {repo_id}")
    print(f"  路径: {local_dir}")
    print(f"  端点: {endpoint}")
    print()

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.png", "*.jpg", "*.pdf", "*.md"],
    )
    print(f"[完成] {desc} -> {local_dir}\n")
    return local_dir


def main():
    parser = argparse.ArgumentParser(description="下载评估所需的模型权重")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["c3", "qwen"],
        default=["c3", "qwen"],
        help="要下载的模型（默认：全部）",
    )
    parser.add_argument(
        "--c3_repo",
        default="liufanfanlff/C3-Context-Cascade-Compression",
        help="C3 模型 HuggingFace 仓库 ID",
    )
    parser.add_argument(
        "--qwen_repo",
        default="Qwen/Qwen2.5-3B",
        help="Qwen2.5-3B HuggingFace 仓库 ID",
    )
    parser.add_argument(
        "--output_dir",
        default="./models",
        help="模型保存根目录（默认：./models）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if "c3" in args.models:
        download(
            repo_id=args.c3_repo,
            local_dir=os.path.join(args.output_dir, "c3"),
            desc="C3-Context-Cascade-Compression (~9.3GB)",
        )

    if "qwen" in args.models:
        download(
            repo_id=args.qwen_repo,
            local_dir=os.path.join(args.output_dir, "qwen25-3b"),
            desc="Qwen2.5-3B (~6GB)",
        )

    print("所有模型下载完成。")
    print(f"  C3 路径:   {os.path.join(args.output_dir, 'c3')}")
    print(f"  Qwen 路径: {os.path.join(args.output_dir, 'qwen25-3b')}")


if __name__ == "__main__":
    main()
