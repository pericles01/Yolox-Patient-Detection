import os
import sys
import argparse
import random
from unicodedata import name
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import yolox
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from azureml.core import Run
from threading import Thread

from log_metrics import log_metrics
from launch import launch
from custom_trainer import CustomTrainer


YOLOX_VERSION = yolox.__version__
TORCH_VERSION = torch.__version__

@logger.catch
def main(exp):

    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = CustomTrainer(exp)
    trainer.train()

    logger.info(f'Emit end of train')
    exp.emit_metric('end', None)


def train(conf, code_prefix = './'):

    #modulename = 'log_metrics'
    #if modulename not in sys.modules:
        #print('You have not imported the {} module'.format(modulename))
        #from src.log_metrics import log_metrics
        #from src.launch import launch

    # Need Azure run context for tagging and logging, passed to Exp
    run = Run.get_context()

    # Loads an instance of Exp class from the file specified
    exp = get_exp(os.path.join(code_prefix, 'main_experiment.py'), None)
    exp.setup(conf)

    run.tag('model_name', conf.get('model_name'))
    run.tag('batch_size', conf.get('batch_size'))
    run.tag('max_epoch', conf.get('max_epoch'))
    run.tag('augment', str(conf.get('augment')))
    run.tag('eval_interval', exp.eval_interval)
    run.tag('config', conf.get_path())
    run.tag('yolox_version', YOLOX_VERSION)
    run.tag('torch_version', TORCH_VERSION)

    logger.info(f'Starting training with {conf.get("model_name")} ..')
    logger.info(f'Input dir {conf.get("indir")}')
    logger.info(f'Output dir {conf.get("outdir")}')
    logger.info(f'Using run config file {conf.get_path()}')

    if conf.get('ckpt') is not None:
        logger.info(f'pretrained with weights: {conf.get("ckpt")}')
        run.tag('pretrained', "True")

    smp = mp.get_context('spawn')
    exp.metrics_queue = smp.SimpleQueue()

    num_gpu = get_num_devices()
    logger.info(f'Number of GPUs: {num_gpu}')

    # Normally little effect, but attempt anyway
    torch.cuda.empty_cache()

    # Starts a thread that listens to metric log events and sends metric to Azure ML
    log_thread = Thread(target = log_metrics, daemon=True, args = (exp.metrics_queue, num_gpu, run,))
    log_thread.start()

    proc_context = launch(main, num_gpu, 1, 0, backend='nccl', dist_url='auto', args=exp)

    if proc_context is not None:
        logger.info('right before proc_context.join()')
        proc_context.join()

    # Thread is terminated by sending 'end' to exp.metrics_queue.
    logger.info('right before log_thread.join()')
    log_thread.join()
    logger.info('after log_thread.join()')

if __name__ == "__main__":
    logger.info(f"YOLOX version: {YOLOX_VERSION}")
    logger.info(f"TORCH version: {TORCH_VERSION}")

    # Need only these to be parsed because they are dynamically created by azureml SDK
    # All options are defined in the yaml file passed with --config
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, help="input data directory")
    parser.add_argument('--outdir', type=str, help="output data directory")
    parser.add_argument('--ckpt', type=str, default=None, help="path to model weight file (for pretrained weights)")
    parser.add_argument('--config', type=str, default='run_config.yml', help="path to config file of the experiment")
    args = parser.parse_args()

    from experiment_config import ExperimentConfig

    # Load other modules here..
    # In other place (in method), only load if not already loaded
    conf = ExperimentConfig(args.config)
    conf.add(args)
    train(conf)
