# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--epochs', type=int, default=None, help='override number of epochs')
    parser.add_argument('--stopping_step', type=int, default=None, help='override early stopping patience')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--checkpoint_interval', type=int, default=None, help='save latest checkpoint every N epochs')
    parser.add_argument('--checkpoint_run_id', type=str, default=None, help='checkpoint filename prefix')
    parser.add_argument('--use_gpu', action='store_true', help='train on GPU if CUDA is available')
    parser.add_argument('--cpu', action='store_true', help='force CPU training')
    parser.add_argument('--loss_type', type=str, default=None, choices=['bpr', 'softmax'],
                        help='MENTOR ranking loss: bpr or sampled softmax')
    parser.add_argument('--num_negatives', type=int, default=None,
                        help='number of negatives per positive for sampled softmax')
    parser.add_argument('--ssm_temp', type=float, default=None, help='sampled-softmax temperature')
    parser.add_argument('--hard_neg_ratio', type=float, default=None,
                        help='optional fraction of content-kNN hard negatives')
    parser.add_argument('--use_feature_adapter', action='store_true', help='enable residual feature adapters')
    parser.add_argument('--use_id_residual', action='store_true', help='enable ID residual scoring')
    parser.add_argument('--seed', type=int, default=None, help='override random seed')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()
    for key in ['epochs', 'stopping_step', 'resume_checkpoint', 'checkpoint_interval', 'checkpoint_run_id',
                'loss_type', 'num_negatives', 'ssm_temp', 'hard_neg_ratio']:
        value = getattr(args, key)
        if value is not None:
            config_dict[key] = value
    if args.cpu:
        config_dict['use_gpu'] = False
    elif args.use_gpu:
        config_dict['use_gpu'] = True
    if args.use_feature_adapter:
        config_dict['use_feature_adapter'] = True
    if args.use_id_residual:
        config_dict['use_id_residual'] = True
    if args.seed is not None:
        config_dict['seed'] = [args.seed]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


