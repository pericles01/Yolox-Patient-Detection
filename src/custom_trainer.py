#!/usr/bin/env python
from yolox.core import Trainer
from yolox.utils.dist import is_main_process
from loguru import logger
import numpy as np
from types import SimpleNamespace


class CustomTrainer(Trainer):
    """
    This class has the primary purpose to get better access to training metrics.
    The average total_loss per epoch (the most important loss) is not calculated in the
    YoloX implementation.
    The total_loss kept in MetricBuffer are the global average total_loss (over all epochs)
    and the total_loss over the last reporting period
    """

    _losses_iteration = []

    def __init__(self, exp):
        super().__init__(exp, exp.get_args())

    def after_epoch(self):

        """
        Take average over all collected metrics in this epoch and emit for logging.
        """

        #logger.info(f'CustomTrainer.after_epoch() local rank: {self.rank}, local_rank: {self.local_rank}')

        avg_loss_epoch = np.array(self._losses_iteration).mean()
        self._losses_iteration = []

        # emitting self.rank is not necessary but helps debugging the metrics queue end
        self.exp.emit_metric('total_loss', [self.rank, avg_loss_epoch])

        # so that the experiment can access the epoch number for log evaluation
        self.exp.epoch = self.epoch

        super().after_epoch()

    def after_iter(self):

        """
        Get the total_loss (or other metrics) from the last iteration (batch).
        """

        # Save total_loss of this iteration
        loss_meter = self.meter.get_filtered_meter("loss")
        losses = loss_meter['total_loss']
        self._losses_iteration.append(losses.latest.item())

        # Call original method
        super().after_iter()

                # This log proofs that the total loss reported from .latest is actually the loss from the last iteration (batch)
                #    - This is because in CustomTrainer, we extract the .latest property.
                #    - We could extract
                #        .avg        - uses _deque            <- reset on .reset() and .clear() <-- called on
                #        .global_avg - uses _total / _count   <- reset on .reset()              <-- called on
                #
                #    - .clear() called <-- MeterBuffer.clear_meters() <-- Trainer.after_iter()
                #    - .reset() called <-- MeterBuffer.reset()        <-- NOT IN Trainer !
                #
                #   Summary:
                #       1 .avg can't be used because the _deque array is cleared after each iteration
                #       2 .global_avg can't be used because it is never cleared and runs over all epochs
                #   ---> Need to create a average total loss for each epoch.
                #        Best place is CustomTrainer:
                #            - in .after_iter() save latest total_loss (DONE)
                #            - in .after_epoch() take average and emit metric (DONE)
                #            - in main thread: take average over all workers (DONE)
                #
                #
                # Received metrics_queue: {'name': 'after_iter-total_loss', 'value': [0, 12.982063293457031]}
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:131 - Got after_iter total loss of worker 0, total_loss: 0
                # 2021-11-14 12:37:26 | INFO     | custom_trainer:18 - CustomTrainer.after_epoch() local rank: 0, local_rank: 0
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:108 - Waiting for metrics...
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:113 - Received metrics_queue: {'name': 'after_iter-total_loss', 'value': [1, 12.747295379638672]}
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:131 - Got after_iter total loss of worker 1, total_loss: 1
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:108 - Waiting for metrics...
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:113 - Received metrics_queue: {'name': 'total_loss', 'value': [1, 12.747295379638672]}
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:121 - Got total_loss 12.747295379638672 from worker 1
                # 2021-11-14 12:37:26.119 | INFO     | __main__:<module>:108 - Waiting for metrics...
                # 2021-11-14 12:37:26.120 | INFO     | __main__:<module>:113 - Received metrics_queue: {'name': 'total_loss', 'value': [0, 12.982063293457031]}
