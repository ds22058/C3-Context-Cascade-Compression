"""
从 C3 模型 safetensors 中提取 decoder 权重，保存为标准 Qwen2ForCausalLM。

做法：直接读取 C3 主模型的 safetensors 文件（不加载 llm1 子目录），
过滤掉 C3 专属组件（model.Q、model.mm_projector），
写入 Qwen2 格式的 config.json 和 safetensors，同时复制 tokenizer 文件。

内存峰值：约 12GB（3B × 2 份：state_dict + model）
运行时间：CPU 约 3-8 分钟，GPU 约 1 分钟

用法：
    python eval/extract_decoder.py
    python eval/extract_decoder.py --c3_path ./models/c3 --output_path ./models/c3_decoder_extracted
"""

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM


# C3 特有组件前缀，提取 decoder 时需排除
C3_EXCLUSIVE_PREFIXES = [
    "model.Q.",
    "model.mm_projector.",
]

# C3 config 中需要透传给 Qwen2Config 的字段
QWEN2_CONFIG_KEYS = [
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "hidden_act",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "tie_word_embeddings",
    "attention_dropout",
    "max_window_layers",
    "use_sliding_window",
    "sliding_window",
    "use_cache",
]


def load_c3_state_dict(c3_path: str) -> dict:
    """仅加载 C3 主模型的 safetensors（跳过 llm1/ 子目录）"""
    shard_files = sorted(glob.glob(os.path.join(c3_path, "model-*.safetensors")))
    if not shard_files:
        single = os.path.join(c3_path, "model.safetensors")
        if os.path.exists(single):
            shard_files = [single]
        else:
            raise FileNotFoundError(
                f"在 {c3_path} 中找不到 safetensors 文件\n"
                "请先运行 python eval/download_models.py"
            )

    print(f"  找到 {len(shard_files)} 个 safetensors 文件")
    state_dict = {}
    for f in shard_files:
        print(f"  加载: {os.path.basename(f)}")
        state_dict.update(load_file(f, device="cpu"))
    return state_dict


def filter_decoder_weights(state_dict: dict) -> dict:
    """过滤掉 C3 专属组件，保留 decoder 权重"""
    decoder_state = {}
    excluded = []
    for k, v in state_dict.items():
        if any(k.startswith(pfx) for pfx in C3_EXCLUSIVE_PREFIXES):
            excluded.append(k)
        else:
            decoder_state[k] = v

    print(f"  总 keys:  {len(state_dict)}")
    print(f"  排除:     {excluded}")
    print(f"  保留:     {len(decoder_state)}")

    # 确保 lm_head.weight 存在（tie_word_embeddings 时可能缺失）
    if "lm_head.weight" not in decoder_state and "model.embed_tokens.weight" in decoder_state:
        print("  补充 lm_head.weight（tie_word_embeddings）")
        decoder_state["lm_head.weight"] = decoder_state["model.embed_tokens.weight"]

    return decoder_state


def build_qwen2_config(c3_cfg: dict) -> Qwen2Config:
    kwargs = {k: c3_cfg[k] for k in QWEN2_CONFIG_KEYS if k in c3_cfg}
    kwargs.setdefault("torch_dtype", "bfloat16")
    return Qwen2Config(**kwargs)


def extract_decoder(c3_path: str, output_path: str) -> None:
    os.makedirs(output_path, exist_ok=True)

    # ── Step 1: 读取 C3 config ──────────────────────────────────────
    config_file = os.path.join(c3_path, "config.json")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"找不到 {config_file}")
    with open(config_file) as f:
        c3_cfg = json.load(f)
    print(f"[1/5] 读取 C3 config: model_type={c3_cfg['model_type']}, "
          f"hidden_size={c3_cfg['hidden_size']}, layers={c3_cfg['num_hidden_layers']}")

    # ── Step 2: 加载主模型 safetensors ─────────────────────────────
    print("\n[2/5] 加载 C3 主模型 safetensors（跳过 llm1）")
    state_dict = load_c3_state_dict(c3_path)

    # ── Step 3: 过滤 decoder 权重 ───────────────────────────────────
    print("\n[3/5] 过滤 decoder 权重")
    decoder_state = filter_decoder_weights(state_dict)
    del state_dict  # 立即释放内存

    # ── Step 4: 创建 Qwen2ForCausalLM 并加载权重 ────────────────────
    print("\n[4/5] 构建 Qwen2ForCausalLM 并加载权重")
    qwen_cfg = build_qwen2_config(c3_cfg)
    model = Qwen2ForCausalLM(qwen_cfg)

    missing, unexpected = model.load_state_dict(decoder_state, strict=False)
    del decoder_state

    if missing:
        # lm_head.weight 在 tie_word_embeddings 时可能缺失，属正常现象
        non_trivial_missing = [k for k in missing if k != "lm_head.weight"]
        if non_trivial_missing:
            print(f"  警告 - 缺失 keys: {non_trivial_missing}")
    if unexpected:
        print(f"  警告 - 多余 keys: {unexpected}")

    model.tie_weights()
    print("  权重加载完成，已 tie_weights()")

    # ── Step 5: 保存模型 + tokenizer ────────────────────────────────
    print(f"\n[5/5] 保存到: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
    del model

    # 优先从 Qwen2.5-3B 复制 tokenizer（标准 Qwen2TokenizerFast，无自定义代码，
    # 不依赖 tiktoken，兼容 Windows）；若尚未下载则从 C3 复制通用文件。
    qwen_path = os.path.join(os.path.dirname(c3_path), "qwen25-3b")
    if os.path.isdir(qwen_path):
        tokenizer_src = qwen_path
        print(f"  tokenizer 来源: Qwen2.5-3B ({qwen_path})")
    else:
        tokenizer_src = c3_path
        print(f"  tokenizer 来源: C3 ({c3_path})  [建议先下载 Qwen2.5-3B 以避免 Windows 兼容性问题]")

    # 只复制通用 tokenizer 文件，跳过 tokenization_qwen.py（会触发 custom code 检查）
    safe_tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    copied = []
    for fname in safe_tokenizer_files:
        src = os.path.join(tokenizer_src, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, fname))
            copied.append(fname)
    print(f"  tokenizer 文件已复制: {copied}")

    print(f"\n提取完成！输出目录: {output_path}")
    saved_files = os.listdir(output_path)
    print("  文件列表:", sorted(saved_files))


def main():
    parser = argparse.ArgumentParser(description="从 C3 模型提取 decoder 权重为 Qwen2ForCausalLM")
    parser.add_argument("--c3_path", default="./models/c3", help="C3 模型本地路径")
    parser.add_argument(
        "--output_path",
        default="./models/c3_decoder_extracted",
        help="输出目录",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.c3_path):
        print(f"错误：C3 模型路径不存在: {args.c3_path}")
        print("请先运行: python eval/download_models.py")
        raise SystemExit(1)

    extract_decoder(args.c3_path, args.output_path)


if __name__ == "__main__":
    main()
