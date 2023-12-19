import os
import warnings

import hydra
import torch
from diffusion_reward.models.video_models.vqdiffusion.data.build import \
    build_dataloader
from diffusion_reward.models.video_models.vqdiffusion.distributed.launch import launch
from diffusion_reward.models.video_models.vqdiffusion.engine.logger import Logger
from diffusion_reward.models.video_models.vqdiffusion.engine.solver import Solver
from diffusion_reward.models.video_models.vqdiffusion.modeling.build import \
    build_model
from diffusion_reward.models.video_models.vqdiffusion.utils.io import load_yaml_config
from diffusion_reward.models.video_models.vqdiffusion.utils.misc import (
    merge_opts_to_config, modify_config_for_debug, seed_everything)

# environment variables
NODE_RANK = os.environ['AZ_BATCHAI_TASK_INDEX'] if 'AZ_BATCHAI_TASK_INDEX' in os.environ else 0
NODE_RANK = int(NODE_RANK)
MASTER_ADDR, MASTER_PORT = os.environ['AZ_BATCH_MASTER_NODE'].split(':') if 'AZ_BATCH_MASTER_NODE' in os.environ else ("127.0.0.1", 29500)
MASTER_PORT = int(MASTER_PORT)
DIST_URL = 'tcp://%s:%s' % (MASTER_ADDR, MASTER_PORT)


@hydra.main(config_path='../diffusion_reward/configs/models/video_models/vqdiffusion', config_name='default')
def main(args):
    args.save_dir = os.path.abspath(os.path.dirname(__file__))
    args.node_rank = NODE_RANK
    args.dist_url = DIST_URL

    if args.seed is not None or args.cudnn_deterministic:
        seed_everything(args.seed, args.cudnn_deterministic)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable ddp.')
        torch.cuda.set_device(args.gpu)
        args.ngpus_per_node = 1
        args.world_size = 1
    else:
        if args.num_node == 1:
            args.dist_url == "auto"
        else:
            assert args.num_node > 1
        args.ngpus_per_node = torch.cuda.device_count()
        args.world_size = args.ngpus_per_node * args.num_node

    launch(main_worker, args.ngpus_per_node, args.num_node, args.node_rank, args.dist_url, args=(args,))


def main_worker(local_rank, args):
    args.local_rank = local_rank
    args.global_rank = args.local_rank + args.node_rank * args.ngpus_per_node

    # load config
    config = args
    config = merge_opts_to_config(config, args.opts)
    if args.debug:
        config = modify_config_for_debug(config)

    # get logger
    logger = Logger(args)

    # get model 
    model = build_model(config, args)
    # print(model)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # get dataloader
    dataloader_info = build_dataloader(config, args)

    # get solver
    solver = Solver(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)

    # resume 
    if args.load_path is not None: # only load the model paramters
        solver.resume(path=args.load_path,
                      # load_model=True,
                      load_optimizer_and_scheduler=False,
                      load_others=False)
    if args.auto_resume:
        solver.resume()

    solver.train()


if __name__ == '__main__':
    main()
