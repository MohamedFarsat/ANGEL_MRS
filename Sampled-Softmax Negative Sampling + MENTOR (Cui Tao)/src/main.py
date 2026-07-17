import os
import argparse
from utils_package.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='MENTOR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--epochs', type=int, default=None, help='number of training epochs')
    parser.add_argument('--stopping_step', type=int, default=None, help='early stopping patience')
    parser.add_argument('--resume_checkpoint', type=str, default=None, help='checkpoint path to resume from')
    parser.add_argument('--checkpoint_interval', type=int, default=None, help='save latest checkpoint every N epochs')
    parser.add_argument('--checkpoint_run_id', type=str, default=None, help='checkpoint filename prefix')
    parser.add_argument('--use_gpu', action='store_true', help='train on GPU if CUDA is available')
    parser.add_argument('--cpu', action='store_true', help='force CPU training')
    parser.add_argument('--use_feature_adapter', action='store_true', help='enable residual feature adapters')
    parser.add_argument('--use_id_residual', action='store_true', help='enable ID residual scoring')
    # [HARD-NEG] override yaml ratio from CLI; 0.0 = original MENTOR random negatives
    parser.add_argument('--hard_neg_ratio', type=float, default=None,
                        help='fraction of negatives sampled from content kNN lookalikes (0.0 disables)')
    # [GRAPH-REFRESH] rebuild item kNN graph from adapted features every N epochs
    parser.add_argument('--mm_adj_refresh_interval', type=int, default=None,
                        help='rebuild mm_adj from adapted features every N epochs (0 disables)')
    # [SSM] ranking loss upgrade
    parser.add_argument('--num_negatives', type=int, default=None,
                        help='negatives sampled per interaction (1 = original BPR)')
    parser.add_argument('--loss_type', type=str, default=None, choices=['bpr', 'softmax'],
                        help="ranking loss: 'bpr' (original) or 'softmax' (sampled softmax over K negatives)")
    parser.add_argument('--ssm_temp', type=float, default=None,
                        help='sampled-softmax temperature (lower = focus on hardest negatives)')
    # [SEED] override the seed list for repeat runs (variance / significance checks)
    parser.add_argument('--seed', type=int, default=None, help='override random seed')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()
    for key in ['epochs', 'stopping_step', 'resume_checkpoint', 'checkpoint_interval', 'checkpoint_run_id']:
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
    # [HARD-NEG]
    if args.hard_neg_ratio is not None:
        config_dict['hard_neg_ratio'] = args.hard_neg_ratio
    # [GRAPH-REFRESH]
    if args.mm_adj_refresh_interval is not None:
        config_dict['mm_adj_refresh_interval'] = args.mm_adj_refresh_interval
    # [SSM]
    if args.num_negatives is not None:
        config_dict['num_negatives'] = args.num_negatives
    if args.loss_type is not None:
        config_dict['loss_type'] = args.loss_type
    if args.ssm_temp is not None:
        config_dict['ssm_temp'] = args.ssm_temp
    # [SEED] must be a list — quick_start iterates config['seed'] as a hyperparameter axis
    if args.seed is not None:
        config_dict['seed'] = [args.seed]

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
