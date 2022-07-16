#!/usr/bin/env python

from loguru import logger
import numpy as np

def log_metrics(metrics_queue, num_gpu, run):

    logger.info("Log Metrics Thread started")
    total_losses = []
    while True:

        logger.info("Waiting for metrics...")
        metrics = metrics_queue.get()
        if metrics['name'] == 'end':
            break

        if metrics['name'] == 'aps':
            run.log_row("AP_0.5:0.95 (val)", x=int(metrics['epoch']), y=metrics['value'][0])
            run.log_row("AP_0.5 (val)", x=int(metrics['epoch']), y=metrics['value'][1])

        if metrics['name'] == 'total_loss':
            logger.info('* Got total_loss ' + str(metrics['value'][1]) + ' from worker ' + str(metrics['value'][0]))
            total_losses.append(metrics['value'][1])

            if len(total_losses) == num_gpu:
                run.log(name='Total Loss', value=np.array(total_losses).mean())
                total_losses = []

    logger.info('Exiting from Thread log_metrics!')
