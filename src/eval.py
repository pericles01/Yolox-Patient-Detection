#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, fuse_model, get_local_rank, get_model_info, setup_logger

from launch import launch
from azureml.core import Run

@logger.catch
def main(exp):

    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
        )

    is_distributed = False

    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True

    rank = get_local_rank()
    file_name = os.path.join(exp.output_dir, exp.experiment_name)

    if rank == 0:
        os.makedirs(file_name, exist_ok=True)

    setup_logger(file_name, distributed_rank=rank, filename="test_log.txt", mode="a")

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    logger.info("Model Structure:\n{}".format(str(model)))

    evaluator = exp.get_evaluator(exp.batch_size, is_distributed, False, False)

    torch.cuda.set_device(rank)
    model.cuda(rank)
    model.eval()

    ckpt_file = exp.ckpt
    logger.info("loading checkpoint from {}".format(ckpt_file))
    loc = "cuda:{}".format(rank)
    ckpt = torch.load(ckpt_file, map_location=loc)
    
    logger.info("Update model dict")
    pretrained_dict = ckpt["model"]

    logger.info("loading checkpoint...")
    model.load_state_dict(pretrained_dict,strict=False)
    logger.info("loaded checkpoint done.")

    if is_distributed:
        model = DDP(model, device_ids=[rank])

    if exp.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    trt_file = None
    decoder = None

    # start evaluate
    stats, summary = evaluator.evaluate(
        model, is_distributed, exp.fp16, trt_file, decoder, exp.test_size
    )
    logger.info("\n" + summary)

    # Logs to Azure should be done in tabular form
    #run.log(name='AP_50_95', value=stats[0])
    logger.info("AP 50:95 = {}".format(stats[0]))
    #run.log(name='AP_50', value=stats[1])
    logger.info("AP 50 = {}".format(stats[1]))


def evaluate(conf, code_prefix = './'):

    # -- Inititialize experiment --
    exp = get_exp(os.path.join(code_prefix, 'main_experiment.py'), None)
    exp.setup(conf)

    logger.info(f'Input dir: {conf.get("indir")}')
    logger.info(f'Output dir: {conf.get("outdir")}')
    logger.info(f'Model name: {conf.get("model_name")}')
    logger.info(f'Model checkpoint: {conf.get("ckpt")}')

    dist_url = "auto"

    num_gpu = 1 # torch.cuda.device_count()
    logger.info(f'Number if GPUs: {num_gpu}')

     #   args=(exp, None, num_gpu, run),
    launch(main, num_gpu, 1, 0, backend='nccl', dist_url='auto', args=exp)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()

    #    exp.nmsthre = args.nms
    #if exp.tsize is not None:
    #    exp.test_size = (args.tsize, args.tsize)


    from experiment_config import ExperimentConfig
    conf = ExperimentConfig(args.config)
    conf.add(args)

    evaluate(conf)
