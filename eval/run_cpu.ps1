# ============================================================
# C3 Decoder 评估流程 - Windows CPU 版本
# 运行前请切换到项目根目录: cd d:\workspace\C3-Context-Cascade-Compression
# 执行: .\eval\run_cpu.ps1
# ============================================================
# 预计总耗时: 5-10 小时（含下载时间和 PPL 计算）
# 峰值内存:   约 12GB RAM（每次只加载一个模型）

$ErrorActionPreference = "Stop"

function Write-Step {
    param($msg)
    Write-Host ""
    Write-Host ("=" * 55) -ForegroundColor Cyan
    Write-Host $msg -ForegroundColor Cyan
    Write-Host ("=" * 55) -ForegroundColor Cyan
}

# ── Step 0: 检查 Python ──────────────────────────────────────
Write-Step "Step 0: 检查 Python 环境"
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误：找不到 python，请确认已激活正确的 conda/venv 环境。" -ForegroundColor Red
    exit 1
}

# ── Step 1: 下载模型 ─────────────────────────────────────────
Write-Step "Step 1: 下载模型 (C3 ~9.3GB + Qwen2.5-3B ~6GB)"
Write-Host "如果访问 HuggingFace 较慢，可先在 PowerShell 中设置镜像："
Write-Host '  $env:HF_ENDPOINT = "https://hf-mirror.com"'
Write-Host ""
python eval/download_models.py
if ($LASTEXITCODE -ne 0) { Write-Host "下载失败，退出。" -ForegroundColor Red; exit 1 }

# ── Step 2: 提取 C3 Decoder ─────────────────────────────────
Write-Step "Step 2: 提取 C3 Decoder 权重 (~3-8 分钟)"
python eval/extract_decoder.py `
    --c3_path      ./models/c3 `
    --output_path  ./models/c3_decoder_extracted
if ($LASTEXITCODE -ne 0) { Write-Host "提取失败，退出。" -ForegroundColor Red; exit 1 }

# ── Step 3: 重建准确率（CPU，10 条样本，约 30-60 分钟）───────────
Write-Step "Step 3: WikiText-2 重建准确率（CPU，10 条样本）"
Write-Host "注：CPU 上生成较慢，仅评估 10 条短样本。如需完整评估建议使用 GPU。"
python eval/eval_reconstruction.py `
    --device       cpu `
    --max_samples  10 `
    --max_tokens   256 `
    --output       ./eval/results/reconstruction.json
if ($LASTEXITCODE -ne 0) { Write-Host "重建评估失败（忽略，继续）" -ForegroundColor Yellow }

# ── Step 4: Perplexity 评估（CPU，限 200 条文本，约 30-90 分钟）─
Write-Step "Step 4: WikiText-2 Perplexity（CPU，快速模式 --max_samples 200）"
Write-Host "若要全量评估（更准确，需 1-3 小时/模型），删去 --max_samples 参数。"
python eval/eval_perplexity.py `
    --device       cpu `
    --max_samples  200 `
    --output       ./eval/results/perplexity.json
if ($LASTEXITCODE -ne 0) { Write-Host "Perplexity 评估失败，退出。" -ForegroundColor Red; exit 1 }

# ── Step 5: 生成质量对比（CPU，约 15-30 分钟）──────────────────
Write-Step "Step 5: 生成质量对比（CPU）"
python eval/eval_generation.py `
    --device          cpu `
    --max_new_tokens  150 `
    --output          ./eval/results/generation.json
if ($LASTEXITCODE -ne 0) { Write-Host "生成评估失败，退出。" -ForegroundColor Red; exit 1 }

# ── Step 6: NLP 基准（CPU，limit=200，约 1-2 小时/模型）────────
Write-Step "Step 6: NLP 基准测试（CPU，ARC-Easy + WinoGrande，各限 200 条）"
Write-Host "注：CPU 上 MMLU/HellaSwag 耗时极长，此处仅跑两个较小的任务。"

Write-Host "`n  [5a] Qwen2.5-3B 基线..."
lm_eval `
    --model       hf `
    --model_args  "pretrained=./models/qwen25-3b" `
    --tasks       arc_easy,winogrande `
    --device      cpu `
    --batch_size  1 `
    --limit       200 `
    --output_path ./eval/results/lm_eval_qwen25_3b_cpu
if ($LASTEXITCODE -ne 0) { Write-Host "lm_eval Qwen 失败（忽略，继续）" -ForegroundColor Yellow }

Write-Host "`n  [5b] C3-Decoder..."
lm_eval `
    --model       hf `
    --model_args  "pretrained=./models/c3_decoder_extracted" `
    --tasks       arc_easy,winogrande `
    --device      cpu `
    --batch_size  1 `
    --limit       200 `
    --output_path ./eval/results/lm_eval_c3_decoder_cpu
if ($LASTEXITCODE -ne 0) { Write-Host "lm_eval C3 失败（忽略，继续）" -ForegroundColor Yellow }

# ── Step 7: 汇总报告 ─────────────────────────────────────────
Write-Step "Step 7: 生成汇总报告"
python eval/analyze_results.py `
    --ppl_file     ./eval/results/perplexity.json `
    --recon_file   ./eval/results/reconstruction.json `
    --lm_eval_qwen ./eval/results/lm_eval_qwen25_3b_cpu `
    --lm_eval_c3   ./eval/results/lm_eval_c3_decoder_cpu `
    --gen_file     ./eval/results/generation.json `
    --output       ./eval/results/summary.md
if ($LASTEXITCODE -ne 0) { Write-Host "汇总失败，退出。" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "全部完成！报告保存在 ./eval/results/summary.md" -ForegroundColor Green
