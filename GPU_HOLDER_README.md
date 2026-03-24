# GPU Holder — 占卡程序

在共享集群上占住 GPU 并维持利用率，防止被调度器回收。

## 快速开始

```bash
# 占满 8 张 H200（默认）
python gpu_holder.py

# 后台运行
nohup python gpu_holder.py > gpu_holder.log 2>&1 &
```

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gpus` | `0,1,2,3,4,5,6,7` | 要占用的 GPU 编号，逗号分隔 |
| `--mem-gb` | `10` | 每张卡占用的显存 (GB) |
| `--mat-dim` | `4096` | 矩阵乘法维度，越大利用率越高 |

## 用法示例

```bash
# 只占 4 张卡
python gpu_holder.py --gpus 0,1,2,3

# 每卡占 20GB 显存
python gpu_holder.py --mem-gb 20

# 提高利用率（更大的矩阵）
python gpu_holder.py --mat-dim 8192

# 降低利用率（更小的矩阵）
python gpu_holder.py --mat-dim 2048
```

## 停止

```bash
# 前台运行 → Ctrl+C

# 后台运行 → kill
kill $(pgrep -f gpu_holder.py)
```

## 原理

每张 GPU 上启动一个独立进程：
1. 分配一块指定大小的显存（`torch.empty`）占住显存
2. 循环执行 float16 矩阵乘法维持 GPU 利用率
3. 收到 SIGINT/SIGTERM 后释放显存退出
