#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import sys
from datetime import timedelta
from loguru import logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import yolox.utils.dist as comm

__all__ = ["launch"]


DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    """
    Find an available port of current machine / node.
    """
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend="nccl",
    dist_url=None,
    args=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        cache = vars(args[1]).get("cache", False)

        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", (
                "As Windows platform doesn't support fork method, "
                "do not add --cache in your training command."
            )
            start_method = "fork"

        proc_context = mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            join=True,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
        
    else:
        main_func(*args)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    main_func(*args)

counter = 0
def main(x, args):
    from random import randint
    from time import sleep
    global counter
    for i in range(1, 11):
        args.metrics_queue.put({"ap50":i})
        print(f"Send:{i}")
    args.metrics_queue.put('')

if __name__ == "__main__":
    metrics_queue = mp.SimpleQueue()
    args = argparse.Namespace(
        metrics_queue = metrics_queue)

    dist_url = "auto"
    launch(
        main,
        4,
        1,
        0,
        backend='nccl',
        dist_url=dist_url,
        args=(0, args),
    )

    print('After launch')


    while True:
        metric = metrics_queue.get()
        print(f'Metric: {metric}')
        if metric == '':
            break
