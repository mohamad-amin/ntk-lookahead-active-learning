import argparse
import os
import shutil
import yaml


def set_seeds(config_path):
    with open(config_path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    seed = config['data'].get('seed', 1313)

    # Fixing the seeds for data
    import numpy
    import random
    numpy.random.seed(seed)
    random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':

    # Fixing the pthread_cancel glitch while using python 3.8 (you can comment these two lines if you're on 3.7)
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='',
                        help="Path to a config")
    parser.add_argument('--save_dir', default='',
                        help='Path to dir to save checkpoints and logs')
    parser.add_argument('--eval_only', action='store_true', default=False,
                        help='run eval only using the given checkpoint path')
    args = parser.parse_args()

    set_seeds(args.config_path)

    os.makedirs(args.save_dir, exist_ok=True)
    from src.utils.util import load_log
    logger = load_log(args.save_dir)
    shutil.copyfile(args.config_path, os.path.join(args.save_dir, "config.yml"))

    from src.engine import Engine
    engine = Engine(config_path=args.config_path, logger=logger, save_dir=args.save_dir)
    if args.eval_only:
        engine.evaluate()
    else:
        engine.run()
