"""
生成质量评估：使用固定 prompt 集测试单个模型的生成表现。

关注点：
  - 文本连贯性与流畅性
  - 知识准确性
  - 是否出现"复读机"倾向
  - 中英文双语能力

用法：
    python eval/eval_generation.py --model_path ./models/qwen25-3b --model_name qwen25-3b
    python eval/eval_generation.py --model_path ./models/c3_decoder_extracted --model_name c3-decoder
    python eval/eval_generation.py --model_path ./models/qwen25-3b --device cpu --max_new_tokens 150
"""

import argparse
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "The theory of general relativity, proposed by Albert Einstein, explains that",
    "In a small village nestled between the mountains, there lived a young girl who",
    'def fibonacci(n):\n    """Calculate the nth Fibonacci number using recursion."""\n',
    "中国的四大发明包括",
    "The key difference between supervised learning and unsupervised learning is",
    "春天来了，小明走出家门，看见",
    "Repeat the following sentence five times: The quick brown fox jumps over the lazy dog. Answer:",
]


def generate_text(model, tokenizer, prompt, device, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return generated.strip(), round(elapsed, 2)


def main():
    parser = argparse.ArgumentParser(description="生成质量评估（单模型）")
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--model_name", default=None, help="模型名称（默认取目录名）")
    parser.add_argument("--output", default=None, help="结果路径（默认 eval/results/<model_name>/generation.json）")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float32"])
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    model_name = args.model_name or os.path.basename(args.model_path.rstrip("/\\"))
    output = args.output or f"./eval/results/{model_name}/generation.json"

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    else:
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"模型: {model_name}")
    print(f"路径: {args.model_path}")
    print(f"设备: {device}  精度: {dtype}  最大生成: {args.max_new_tokens} tokens")

    if not os.path.isdir(args.model_path):
        print(f"错误：模型路径不存在: {args.model_path}")
        raise SystemExit(1)

    # 加载模型
    print(f"\n加载模型...")
    t0 = time.time()
    if device.type == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, device_map={"": device}, low_cpu_mem_usage=True,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    print(f"  加载耗时: {time.time() - t0:.1f}s")

    # 生成
    results = []
    for i, prompt in enumerate(PROMPTS):
        print(f"\n  [Prompt {i+1}/{len(PROMPTS)}] {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        text, elapsed = generate_text(model, tokenizer, prompt, device, args.max_new_tokens)
        print(f"  耗时: {elapsed}s")
        print(f"  生成: {text[:120]}{'...' if len(text) > 120 else ''}")
        results.append({
            "prompt_id": i + 1,
            "prompt": prompt,
            "generated": text,
            "elapsed_s": elapsed,
        })

    output_data = {
        "model_name": model_name,
        "model_path": args.model_path,
        "device": str(device),
        "dtype": str(dtype),
        "max_new_tokens": args.max_new_tokens,
        "results": results,
    }

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
