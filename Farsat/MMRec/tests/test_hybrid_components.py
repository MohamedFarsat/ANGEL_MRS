import os
import sys
import tempfile
import unittest

import numpy as np
import scipy.sparse as sp
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hybrid_static import blended_scores
from models.mmgf import MMGF


class Config(dict):
    def __getitem__(self, key):
        return self.get(key)


class TinyDataset:
    uid_field = 'userID'
    iid_field = 'itemID'

    def __init__(self):
        rows = np.array([0, 0, 1, 2, 2])
        cols = np.array([0, 1, 1, 2, 3])
        data = np.ones_like(rows, dtype=np.float32)
        self.matrix = sp.coo_matrix((data, (rows, cols)), shape=(3, 4))

    def get_user_num(self):
        return 3

    def get_item_num(self):
        return 4

    def inter_matrix(self, form='coo'):
        return self.matrix.tocsr() if form == 'csr' else self.matrix.tocoo()


class TinyLoader:
    def __init__(self):
        self.dataset = TinyDataset()

    def inter_matrix(self, form='coo'):
        return self.dataset.inter_matrix(form=form)


class FixedScoreModel:
    def __init__(self, scores):
        self.scores = scores

    def full_sort_predict(self, interaction):
        users = interaction[0]
        return self.scores[users]


class HybridComponentTests(unittest.TestCase):
    def test_mmgf_outputs_finite_full_sort_scores(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = os.path.join(tmpdir, 'tiny')
            os.makedirs(dataset_dir)
            np.save(os.path.join(dataset_dir, 'image_feat.npy'), np.random.RandomState(1).randn(4, 5).astype(np.float32))
            np.save(os.path.join(dataset_dir, 'text_feat.npy'), np.random.RandomState(2).randn(4, 3).astype(np.float32))

            config = Config({
                'USER_ID_FIELD': 'userID',
                'ITEM_ID_FIELD': 'itemID',
                'NEG_PREFIX': 'neg_',
                'train_batch_size': 2,
                'device': torch.device('cpu'),
                'end2end': False,
                'is_multimodal_model': True,
                'data_path': tmpdir + os.sep,
                'dataset': 'tiny',
                'vision_feature_file': 'image_feat.npy',
                'text_feature_file': 'text_feat.npy',
                'mmgf_knn_k': 2,
                'mmgf_coefficients': [0.2, 0.1, -0.1],
                'mmgf_beta': 0.5,
                'mmgf_gamma': 0.5,
            })
            model = MMGF(config, TinyLoader())
            scores = model.full_sort_predict([torch.tensor([0, 2])])
            self.assertEqual(tuple(scores.shape), (2, 4))
            self.assertTrue(torch.isfinite(scores).all())

    def test_blend_extremes_match_component_rankings_after_zscore(self):
        mentor_scores = torch.tensor([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]])
        mmgf_scores = torch.tensor([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
        users = torch.tensor([0, 1])
        empty_mask = torch.empty((2, 0), dtype=torch.long)
        interaction = [users, empty_mask]
        mentor = FixedScoreModel(mentor_scores)
        mmgf = FixedScoreModel(mmgf_scores)

        mmgf_only = blended_scores(mentor, mmgf, interaction, 0.0)
        mentor_only = blended_scores(mentor, mmgf, interaction, 1.0)

        self.assertTrue(torch.equal(torch.argsort(mmgf_only, dim=1, descending=True),
                                    torch.argsort(mmgf_scores, dim=1, descending=True)))
        self.assertTrue(torch.equal(torch.argsort(mentor_only, dim=1, descending=True),
                                    torch.argsort(mentor_scores, dim=1, descending=True)))


if __name__ == '__main__':
    unittest.main()
