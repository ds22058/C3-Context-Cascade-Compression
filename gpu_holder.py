#!/usr/bin/env python3
"""
GPU Holder — 占卡并维持 GPU 利用率
"""

import argparse
import os
import signal
import time
from multiprocessing import Process, Event

import torch


def hold_gpu(device_id: int, stop_event: Event, mem_gb: float, mat_dim: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    torch.cuda.set_device(0)
    dev = torch.device("cuda:0")

    placeholder = torch.empty(int(mem_gb * 1024 ** 3) // 4, dtype=torch.float32, device=dev)
    a = torch.randn(mat_dim, mat_dim, device=dev, dtype=torch.float16)
    b = torch.randn(mat_dim, mat_dim, device=dev, dtype=torch.float16)

    print(f"[GPU {device_id}] 占用 {mem_gb:.1f} GB 显存, 矩阵维度 {mat_dim}")

    while not stop_event.is_set():
        try:
            torch.mm(a, b)
            torch.cuda.synchronize(dev)
            time.sleep(0.02)
        except KeyboardInterrupt:
            break

    del placeholder, a, b
    torch.cuda.empty_cache()
    print(f"[GPU {device_id}] 已释放")


def main():
    parser = argparse.ArgumentParser(description="GPU 占卡程序")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="GPU 编号，逗号分隔 (默认 0-7)")
    parser.add_argument("--mem-gb", type=float, default=10.0,
                        help="每卡占用显存 GB (默认 10)")
    parser.add_argument("--mat-dim", type=int, default=4096,
                        help="矩阵乘法维度，越大利用率越高 (默认 4096)")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
    stop_event = Event()

    signal.signal(signal.SIGINT, lambda *_: stop_event.set())
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())

    print(f"占卡: GPU {gpu_ids} | 显存 {args.mem_gb} GB/卡 | 矩阵 {args.mat_dim}x{args.mat_dim}")
    print("Ctrl+C 或 kill 退出\n")

    procs = []
    for gid in gpu_ids:
        p = Process(target=hold_gpu, args=(gid, stop_event, args.mem_gb, args.mat_dim), daemon=True)
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
