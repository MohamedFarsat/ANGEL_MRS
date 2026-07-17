# coding: utf-8

import argparse
import copy
import os
from logging import getLogger
from time import time

import numpy as np
import torch

from common.trainer import Trainer
from utils.configurator import Config
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.metrics import metrics_dict
from utils.utils import init_seed, get_model, early_stopping, dict2str


REPORT_METRICS = ['Recall', 'NDCG']
REPORT_TOPK = [10, 20]


def parse_weight_grid(raw):
    if raw:
        return [float(x) for x in raw.split(',')]
    return [round(i * 0.05, 2) for i in range(21)]


def materialize_first_hyperparams(config):
    for key in config['hyper_parameters']:
        value = config[key]
        if isinstance(value, list):
            config[key] = value[0]
    if isinstance(config['seed'], list):
        config['seed'] = config['seed'][0]


def build_data(config):
    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    for split_dataset in (train_dataset, valid_dataset, test_dataset):
        split_dataset.inter_num = len(split_dataset)
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'], shuffle=True)
    valid_data = EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset,
                                batch_size=config['eval_batch_size'])
    test_data = EvalDataLoader(config, test_dataset, additional_dataset=train_dataset,
                               batch_size=config['eval_batch_size'])
    return dataset, train_data, valid_data, test_data


def clone_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_with_best_restore(config, model, train_data, valid_data):
    logger = getLogger()
    trainer = Trainer(config, model)
    best_state = clone_state_dict(model)
    best_valid_score = -1
    best_valid_result = None
    cur_step = 0

    for epoch_idx in range(trainer.start_epoch, trainer.epochs):
        start_time = time()
        model.pre_epoch_processing()
        train_loss, _ = trainer._train_epoch(train_data, epoch_idx)
        if torch.is_tensor(train_loss):
            logger.info('Stopping because training returned a tensor loss sentinel.')
            break
        trainer.lr_scheduler.step()
        post_info = model.post_epoch_processing()
        logger.info(trainer._generate_train_loss_output(epoch_idx, start_time, time(), train_loss))
        if post_info is not None:
            logger.info(post_info)

        if (epoch_idx + 1) % trainer.eval_step != 0:
            continue

        valid_score, valid_result = trainer._valid_epoch(valid_data)
        best_valid_score, cur_step, stop_flag, update_flag = early_stopping(
            valid_score, best_valid_score, cur_step,
            max_step=trainer.stopping_step, bigger=trainer.valid_metric_bigger
        )
        logger.info('epoch %d validation: %s' % (epoch_idx, dict2str(valid_result)))
        if update_flag:
            best_state = clone_state_dict(model)
            best_valid_result = valid_result
            logger.info('%s best validation updated.' % config['model'])
        if stop_flag:
            logger.info('Early stopping at epoch %d.' % epoch_idx)
            break

    model.load_state_dict(best_state)
    return trainer, best_valid_score, best_valid_result


def zscore_rows(scores):
    mean = scores.mean(dim=1, keepdim=True)
    std = scores.std(dim=1, keepdim=True).clamp_min(1e-8)
    return (scores - mean) / std


@torch.no_grad()
def blended_scores(trained_model, mmgf_model, batched_data, weight, calibration='zscore'):
    trained_scores = trained_model.full_sort_predict(batched_data)
    mmgf_scores = mmgf_model.full_sort_predict(batched_data)
    if calibration == 'zscore':
        trained_scores = zscore_rows(trained_scores)
        mmgf_scores = zscore_rows(mmgf_scores)
    elif calibration != 'raw':
        raise ValueError('Unsupported blend calibration: %s' % calibration)
    scores = weight * trained_scores + (1.0 - weight) * mmgf_scores
    masked_items = batched_data[1]
    scores[masked_items[0], masked_items[1]] = -1e10
    return scores


@torch.no_grad()
def collect_hybrid_topk(trained_model, mmgf_model, eval_data, config, weight, max_k=20):
    trained_model.eval()
    mmgf_model.eval()
    batch_matrix_list = []
    for batched_data in eval_data:
        scores = blended_scores(
            trained_model, mmgf_model, batched_data, weight,
            calibration=config['blend_calibration'] or 'zscore'
        )
        _, topk_index = torch.topk(scores, max_k, dim=-1)
        batch_matrix_list.append(topk_index.cpu())
    return torch.cat(batch_matrix_list, dim=0).numpy()


def result_from_topk(topk_index, eval_data, metrics=REPORT_METRICS, topk=REPORT_TOPK,
                     user_mask=None, item_mask=None):
    pos_items = eval_data.get_eval_items()
    pos_len_list = eval_data.get_eval_len_list()
    eval_users = eval_data.get_eval_users().numpy()

    bool_rec_matrix = []
    filtered_pos_len = []
    for row_idx, (items, pos_len) in enumerate(zip(pos_items, pos_len_list)):
        user = eval_users[row_idx]
        if user_mask is not None and not user_mask[user]:
            continue
        if item_mask is not None:
            item_set = set(int(i) for i in items if item_mask[int(i)])
            if not item_set:
                continue
        else:
            item_set = set(int(i) for i in items)
            if pos_len <= 0:
                continue
        bool_rec_matrix.append([item in item_set for item in topk_index[row_idx]])
        filtered_pos_len.append(len(item_set))

    if not bool_rec_matrix:
        return {'{}@{}'.format(metric, k): 0.0 for metric in metrics for k in topk}

    bool_rec_matrix = np.asarray(bool_rec_matrix)
    filtered_pos_len = np.asarray(filtered_pos_len)
    metric_dict = {}
    for metric in metrics:
        values = metrics_dict[metric.lower()](bool_rec_matrix, filtered_pos_len)
        for k in topk:
            metric_dict['{}@{}'.format(metric, k)] = round(float(values[k - 1]), 4)
    return metric_dict


def degree_bucket_masks(train_data):
    matrix = train_data.inter_matrix(form='csr')
    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    item_degree = np.asarray(matrix.sum(axis=0)).ravel()
    return {
        'user<=5': user_degree <= 5,
        'user6-20': (user_degree > 5) & (user_degree <= 20),
        'user>20': user_degree > 20,
    }, {
        'item<=5': item_degree <= 5,
        'item6-20': (item_degree > 5) & (item_degree <= 20),
        'item>20': item_degree > 20,
    }


def evaluate_hybrid(trained_model, mmgf_model, eval_data, train_data, config, weight):
    topk_index = collect_hybrid_topk(trained_model, mmgf_model, eval_data, config, weight, max_k=max(REPORT_TOPK))
    result = {'overall': result_from_topk(topk_index, eval_data)}
    user_masks, item_masks = degree_bucket_masks(train_data)
    for name, mask in user_masks.items():
        result[name] = result_from_topk(topk_index, eval_data, user_mask=mask)
    for name, mask in item_masks.items():
        result[name] = result_from_topk(topk_index, eval_data, item_mask=mask)
    return result


def tune_weight(trained_model, mmgf_model, valid_data, config, weight_grid):
    best_weight = weight_grid[0]
    best_score = -1
    best_result = None
    for weight in weight_grid:
        topk_index = collect_hybrid_topk(trained_model, mmgf_model, valid_data, config, weight, max_k=max(REPORT_TOPK))
        result = result_from_topk(topk_index, valid_data)
        score = result[config['blend_valid_metric'] or 'Recall@20']
        getLogger().info('blend weight %.2f validation: %s' % (weight, dict2str(result)))
        if score > best_score:
            best_weight = weight
            best_score = score
            best_result = result
    return best_weight, best_score, best_result


def run(args):
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    weight_grid = parse_weight_grid(args.weight_grid)
    config_dict = {
        'gpu_id': args.gpu_id,
        'use_gpu': not args.no_cuda,
        'save_recommended_topk': False,
        'metrics': REPORT_METRICS,
        'topk': REPORT_TOPK,
        'valid_metric': 'Recall@20',
        'blend_weight_grid': weight_grid,
        'blend_calibration': args.blend_calibration,
        'blend_valid_metric': args.blend_valid_metric,
    }
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
    if args.eval_batch_size is not None:
        config_dict['eval_batch_size'] = args.eval_batch_size
    if args.train_batch_size is not None:
        config_dict['train_batch_size'] = args.train_batch_size

    config = Config(args.trained_model, args.dataset, config_dict)
    materialize_first_hyperparams(config)
    init_logger(config)
    logger = getLogger()
    init_seed(config['seed'])

    logger.info(config)
    _, train_data, valid_data, test_data = build_data(config)
    train_data.pretrain_setup()

    trained_model = get_model(args.trained_model)(config, train_data).to(config['device'])
    logger.info(trained_model)
    _, best_valid_score, best_valid_result = train_with_best_restore(config, trained_model, train_data, valid_data)
    logger.info('Best trained-model validation score: %s, result: %s' %
                (best_valid_score, dict2str(best_valid_result or {})))

    mmgf_model = get_model(args.mmgf_model)(config, train_data).to(config['device'])
    logger.info(mmgf_model)

    best_weight, best_blend_score, best_blend_valid = tune_weight(
        trained_model, mmgf_model, valid_data, config, weight_grid
    )
    logger.info('Best blend weight: %.2f, validation score: %.4f, result: %s' %
                (best_weight, best_blend_score, dict2str(best_blend_valid)))

    test_result = evaluate_hybrid(trained_model, mmgf_model, test_data, train_data, config, best_weight)
    logger.info('Hybrid test results by bucket:')
    for bucket, result in test_result.items():
        logger.info('%s: %s' % (bucket, dict2str(result)))
    return best_weight, best_blend_valid, test_result


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='baby')
    parser.add_argument('--trained_model', type=str, default='MENTOR')
    parser.add_argument('--mmgf_model', type=str, default='MMGF')
    parser.add_argument('--weight_grid', type=str, default='')
    parser.add_argument('--blend_calibration', type=str, default='zscore')
    parser.add_argument('--blend_valid_metric', type=str, default='Recall@20')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--train_batch_size', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=None)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--no_cuda', action='store_true')
    return parser


if __name__ == '__main__':
    run(build_arg_parser().parse_args())
