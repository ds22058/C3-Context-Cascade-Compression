#!/usr/bin/env python3
"""
GPU Holder — 占显存 + 刷 GPU 利用率

用法:
  python gpu_holder.py --mem-gb 120                    # 占 120GB 显存, 利用率拉满
  python gpu_holder.py --mem-gb 120 --util 30          # 占 120GB 显存, 利用率 ~30%
  python gpu_holder.py --mem-gb 120 --util 0           # 只占显存, 不刷利用率
  python gpu_holder.py --mem-gb 120 --gpus 0,1,2,3     # 只占 4 张卡
"""

import argparse
import os
import signal
import time
from multiprocessing import Process, Event

import torch


def hold_gpu(device_id: int, stop_event: Event, mem_gb: float,
             mat_dim: int, util_pct: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    placeholder = torch.empty(
        int(mem_gb * 1024**3) // 4, dtype=torch.float32, device=dev
    )

    if util_pct <= 0:
        print(f"[GPU {device_id}] 占用 {mem_gb:.1f} GB 显存 (仅占显存, 不刷利用率)")
        while not stop_event.is_set():
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                break
        del placeholder
        torch.cuda.empty_cache()
        print(f"[GPU {device_id}] 已释放")
        return

    a = torch.randn(mat_dim, mat_dim, device=dev, dtype=torch.float16)
    b = torch.randn(mat_dim, mat_dim, device=dev, dtype=torch.float16)

    # 通过 duty-cycle 控制利用率: 计算 work_ms 再 sleep rest_ms
    cycle_ms = 100.0
    work_ms = cycle_ms * min(util_pct, 100) / 100.0
    rest_ms = cycle_ms - work_ms

    print(f"[GPU {device_id}] 占用 {mem_gb:.1f} GB, 矩阵 {mat_dim}, "
          f"目标利用率 ~{util_pct}% (work {work_ms:.0f}ms / sleep {rest_ms:.0f}ms)")

    while not stop_event.is_set():
        try:
            t0 = time.monotonic()
            while (time.monotonic() - t0) * 1000 < work_ms:
                torch.mm(a, b)
            torch.cuda.synchronize(dev)

            if rest_ms > 0:
                time.sleep(rest_ms / 1000.0)
        except KeyboardInterrupt:
            break

    del placeholder, a, b
    torch.cuda.empty_cache()
    print(f"[GPU {device_id}] 已释放")


def main():
    parser = argparse.ArgumentParser(description="GPU 占卡程序 (显存 + 利用率)")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="GPU 编号，逗号分隔 (默认 0-7)")
    parser.add_argument("--mem-gb", type=float, default=10.0,
                        help="每卡占用显存 GB (默认 10)")
    parser.add_argument("--mat-dim", type=int, default=4096,
                        help="矩阵乘法维度 (默认 4096)")
    parser.add_argument("--util", type=int, default=100,
                        help="目标 GPU 利用率 %% (0=仅占显存, 100=拉满, 默认 100)")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    stop_event = Event()

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    print(f"占卡: GPU {gpu_ids} | 显存 {args.mem_gb} GB/卡 | "
          f"矩阵 {args.mat_dim}x{args.mat_dim} | 目标利用率 {args.util}%")
    print("Ctrl+C 或 kill 退出\n")

    procs = []
    for gid in gpu_ids:
        p = Process(
            target=hold_gpu,
            args=(gid, stop_event, args.mem_gb, args.mat_dim, args.util),
            daemon=True,
        )
        p.start()
        procs.append(p)

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()

    for p in procs:
        p.join(timeout=10)
    print("全部释放，退出")


if __name__ == "__main__":
    main()
