#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import random
from unicodedata import name
from types import SimpleNamespace

from loguru import logger

import torch
import torch.distributed as dist
import torch.nn as nn

from yolox.exp import Exp as MyExp
from yolox.utils.dist import is_main_process

from yolox.data import COCODataset, ValTransform
from yolox.data import (TrainTransform, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection, worker_init_reset_seed)
from yolox.utils import (wait_for_the_master, get_local_rank, get_num_devices)

from coco_evaluator import COCOEvaluator

import numpy as np # for mean()


class Exp(MyExp):
    def __init__(self):
        super().__init__()
        # Can't call constructor directly and pass some args
        # This is done by get_ext() in yolox/exp/build.py
        # For this reason we have self.setup() below.
        self.num_classes = 1
        self.eval_interval = 2
        self.epoch = 0          # incremented by trainer class
        self.depth = 1.0        # default: yolox-l
        self.width = 1.0        # default: yolox-l
        self.data_dir = None
        self.metrics_queue = None
        self.fuse = False
        self.print_interval = 10
        self.train_subdir = 'images'
        self.val_subdir = "images"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.data_num_workers = get_num_devices()

    def setup(self, conf):

        self.data_dir = conf.get('indir')
        self.output_dir = conf.get('outdir')
        self.ckpt = conf.get('ckpt', None)
        self.test_conf = conf.get('conf', 0.2)
        self.nmsthre = conf.get('nmsthre', 0.5)
        self.num_classes = conf.get('num_classes', self.num_classes)
        self.experiment_name = conf.get('experiment')
        self.exp_name = conf.get('model_name', 'yolox-s')
        self.batch_size = conf.get('batch_size', 16)
        self.max_epoch = conf.get('max_epoch', self.max_epoch)
        self.augment = conf.get('augment', False)
        self.mosaic_enabled = conf.get('mosaic_enabled', False)
        self.cache = conf.get('cache', False)
        self.resume = conf.get('resume', False)
        self.occupy = conf.get('occupy', False)
        self.fp16 = conf.get('fp16', False)
        self.val_ann = conf.get('val_ann', self.val_ann)
        self.val_subdir = conf.get('val_subdir', self.val_subdir)

        self.setup_res(self.exp_name)

    def setup_res(self, model_name):

        # Params: 8.94M, Gflops: 26.64
        if model_name == 'yolox-s':
            self.depth = 0.33
            self.width = 0.50

        elif model_name == 'yolox-m':
            self.depth = 0.67
            self.width = 0.75

    def get_args(self):
        return SimpleNamespace(
            fp16 = self.fp16,
            experiment_name = self.experiment_name,
            batch_size = self.batch_size,
            cache = self.cache,
            resume = self.resume,
            ckpt = self.ckpt,
            occupy = self.occupy
        )

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):

        #logger.info(f'get_eval_loader: annotations in {self.data_dir}/annotations/{self.val_ann}')
        #logger.info(f'get_eval_loader: images in {self.data_dir}/{self.val_subdir}')

        valdataset = COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann,
            name=self.val_subdir,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):

        logger.info(f'get_data_loader: annotations in {self.data_dir}/annotations/{self.train_ann}')
        logger.info(f'get_data_loader: images in {self.data_dir}/{self.train_subdir}')

        local_rank = get_local_rank()
        with wait_for_the_master(local_rank):

            dataset = COCODataset(
                name=self.train_subdir,
                data_dir=self.data_dir,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )


        if self.augment:
            logger.info('Augmentation enabled with params:')
            logger.info('  mosaic_enabled: ' + str(self.mosaic_enabled))
            logger.info('  mosaic_prob: ' + str(self.mosaic_prob))
            logger.info('  mosaic_scale: ' + str(self.mosaic_scale))
            logger.info('  degrees: ' + str(self.degrees))
            logger.info('  translate: ' + str(self.translate))
            logger.info('  enable_mixup: ' + str(self.enable_mixup))
            logger.info('  mixup_prob: ' + str(self.mixup_prob))
            logger.info('  mixup_scale: ' + str(self.mixup_scale))
            logger.info('  shear: ' + str(self.shear))
            logger.info('  preproc - flip_prob: ' + str(self.flip_prob))
            logger.info('  preproc - hsv_prob: ' + str(self.hsv_prob))

            dataset = MosaicDetection(
                dataset,
                img_size=self.input_size,
                mosaic=self.mosaic_enabled,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
            )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=self.augment and self.mosaic_enabled
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):

        logger.info(f'get custom evaluator')
        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        logger.info(f'test_conf: ' + str(self.test_conf))
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )

        return evaluator

    def emit_metric(self, name, metrics):
        metric = {
            'name': name,
            'value': metrics,
            'epoch': self.epoch
        }

        self.metrics_queue.put(metric)

    def eval(self, model, evaluator, is_distributed, half=False):

        cocoEval, summary = evaluator.evaluate(model, is_distributed, half)

        if cocoEval is None:
            return 0,0,summary

        if is_main_process():
            self.emit_metric('aps', cocoEval)

        return cocoEval[0], cocoEval[1], summary
