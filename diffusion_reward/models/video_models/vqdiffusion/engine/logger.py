from __future__ import absolute_import, division, print_function

import os
import time

import torch

from ..distributed.distributed import is_primary
from ..utils.io import save_config_to_yaml, write_args


class Logger(object):
    def __init__(self, args):
        self.args = args
        self.save_dir = args.save_dir
        self.is_primary = is_primary()
        
        if self.is_primary:
            # save the args and config
            self.config_dir = 'configs'
            os.makedirs(self.config_dir, exist_ok=True)
            file_name = os.path.join(self.config_dir, 'args.txt')
            write_args(args, file_name)

            log_dir = 'logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            self.text_writer = open(os.path.join(log_dir, 'log.txt'), 'a') # 'w')
            if args.tensorboard:
                self.log_info('using tensorboard')
                self.tb_writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir) # tensorboard.SummaryWriter(log_dir=log_dir)
            else:
                self.tb_writer = None
            

    def save_config(self, config):
        if self.is_primary:
            save_config_to_yaml(config, os.path.join(self.config_dir, 'config.yaml'))

    def log_info(self, info, check_primary=True):
        if self.is_primary or (not check_primary):
            print(info)
            if self.is_primary:
                info = str(info)
                time_str = time.strftime('%Y-%m-%d-%H-%M')
                info = '{}: {}'.format(time_str, info)
                if not info.endswith('\n'):
                    info += '\n'
                self.text_writer.write(info)
                self.text_writer.flush()

    def add_scalar(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalar(**kargs)

    def add_scalars(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_scalars(**kargs)

    def add_image(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_image(**kargs)

    def add_images(self, **kargs):
        """Log a scalar variable."""
        if self.is_primary:
            if self.tb_writer is not None:
                self.tb_writer.add_images(**kargs)


    def close(self):
        if self.is_primary:
            self.text_writer.close()
            self.tb_writer.close()

