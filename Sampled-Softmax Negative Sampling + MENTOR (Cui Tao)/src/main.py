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
    parser.add_argument('--valid_metric', type=str, default=None,
                        help='validation metric used for early stopping/checkpoint selection, e.g. Recall@20')
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
    parser.add_argument('--hard_bpr_weight', type=float, default=None,
                        help='auxiliary BPR weight against the highest-scoring sampled negative')
    # [POP-NEG] popularity-balanced negative sampling
    parser.add_argument('--pop_neg_ratio', type=float, default=None,
                        help='probability of sampling a negative from popularity buckets instead of uniform random')
    parser.add_argument('--pop_head_weight', type=float, default=None,
                        help='bucket weight for head/popular item negatives')
    parser.add_argument('--pop_mid_weight', type=float, default=None,
                        help='bucket weight for mid-popularity item negatives')
    parser.add_argument('--pop_tail_weight', type=float, default=None,
                        help='bucket weight for tail/rare item negatives')
    # [SEED] override the seed list for repeat runs (variance / significance checks)
    parser.add_argument('--seed', type=int, default=None, help='override random seed')
    # [DNS] dynamic hard-negative selection within the sampled pool
    parser.add_argument('--dns_keep', type=int, default=None,
                        help='keep only K selected negatives from the sampled pool (0 = use all)')
    parser.add_argument('--dns_random_keep', type=int, default=None,
                        help='for mixed_semi_hard, keep this many random negatives in addition to dns_keep')
    parser.add_argument('--dns_strategy', type=str, default=None,
                        choices=['hard', 'semi_hard', 'mixed_semi_hard'],
                        help="negative selection strategy inside the sampled pool")
    parser.add_argument('--dns_veto', action='store_true',
                        help='exclude the positive item\'s content-kNN lookalikes from hard-negative selection')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()
    for key in ['epochs', 'stopping_step', 'resume_checkpoint', 'checkpoint_interval', 'checkpoint_run_id', 'valid_metric']:
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
    if args.hard_bpr_weight is not None:
        config_dict['hard_bpr_weight'] = args.hard_bpr_weight
    # [POP-NEG]
    for key in ['pop_neg_ratio', 'pop_head_weight', 'pop_mid_weight', 'pop_tail_weight']:
        value = getattr(args, key)
        if value is not None:
            config_dict[key] = value
    # [SEED] must be a list — quick_start iterates config['seed'] as a hyperparameter axis
    if args.seed is not None:
        config_dict['seed'] = [args.seed]
    # [DNS]
    if args.dns_keep is not None:
        config_dict['dns_keep'] = args.dns_keep
    if args.dns_random_keep is not None:
        config_dict['dns_random_keep'] = args.dns_random_keep
    if args.dns_strategy is not None:
        config_dict['dns_strategy'] = args.dns_strategy
    if args.dns_veto:
        config_dict['dns_veto'] = True

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)
