# coding: utf-8
#
# Paper-derived implementation of MM-GF:
# "Training-free Adjustable Polynomial Graph Filtering for Ultra-fast
# Multimodal Recommendation", arXiv:2503.04406.

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from common.abstract_recommender import GeneralRecommender


class MMGF(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MMGF, self).__init__(config, dataset)
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.alpha = _value_or_default(config['mmgf_alpha'], 0.7)
        self.adjustment = _value_or_default(config['mmgf_adjustment'], 0.6)
        self.knn_k = _value_or_default(config['mmgf_knn_k'], 20)
        self.eps = 1e-12
        self.interaction_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
        self.score_matrix = None

        coeffs, beta, gamma = self._resolve_coefficients(config)
        self.coefficients = coeffs
        self.beta = beta
        self.gamma = gamma

        self.p_mm = self._build_filter()

    def _resolve_coefficients(self, config):
        defaults = {
            'baby': ([-0.1, 1.1, -0.7], 0.1, 0.0),
            'sports': ([1.5, 1.8, 0.7], 1.9, 0.5),
            'clothing': ([-0.1, 0.7, 1.9], 0.5, 0.2),
        }
        coeffs, beta, gamma = defaults.get(config['dataset'], ([-0.1, 1.1, -0.7], 0.1, 0.0))
        if config['mmgf_coefficients'] is not None:
            coeffs = list(config['mmgf_coefficients'])
        if config['mmgf_beta'] is not None:
            beta = config['mmgf_beta']
        if config['mmgf_gamma'] is not None:
            gamma = config['mmgf_gamma']
        return [float(x) for x in coeffs], float(beta), float(gamma)

    def _build_filter(self):
        p_ui = self._build_interaction_graph()
        p_ui_f = self._polynomial_filter(p_ui, self.coefficients)

        p_mm = p_ui_f
        if self.t_feat is not None and self.beta != 0:
            p_txt = self._build_feature_graph(self.t_feat.detach().cpu())
            p_mm = p_mm + self.beta * self._linear_filter(p_txt)
        if self.v_feat is not None and self.gamma != 0:
            p_img = self._build_feature_graph(self.v_feat.detach().cpu())
            p_mm = p_mm + self.gamma * self._linear_filter(p_img)
        return p_mm.tocsr().astype(np.float32)

    def _build_interaction_graph(self):
        r = self.interaction_matrix
        row_degree = np.asarray(r.sum(axis=1)).ravel()
        col_degree = np.asarray(r.sum(axis=0)).ravel()
        left = _safe_power(row_degree, -self.alpha)
        right = _safe_power(col_degree, self.alpha - 1.0)
        r_tilde = sp.diags(left).dot(r).dot(sp.diags(right))
        p = r_tilde.T.dot(r_tilde).tocsr()
        p.data = np.power(np.maximum(p.data, 0), self.adjustment)
        return p

    def _build_feature_graph(self, features):
        knn = self._topk_binary_cosine(features.float(), self.knn_k)
        row_degree = np.asarray(knn.sum(axis=1)).ravel()
        col_degree = np.asarray(knn.sum(axis=0)).ravel()
        left = _safe_power(row_degree, -self.alpha)
        right = _safe_power(col_degree, self.alpha - 1.0)
        s_tilde = sp.diags(left).dot(knn).dot(sp.diags(right))
        p = s_tilde.dot(s_tilde.T).tocsr()
        p.data = np.power(np.maximum(p.data, 0), self.adjustment)
        return p

    def _topk_binary_cosine(self, features, topk):
        features = torch.nn.functional.normalize(features, dim=1)
        n_items = features.shape[0]
        topk = min(topk, n_items)
        rows, cols = [], []
        block_size = 512
        for start in range(0, n_items, block_size):
            end = min(start + block_size, n_items)
            sim = torch.matmul(features[start:end], features.t())
            _, idx = torch.topk(sim, topk, dim=1)
            row = torch.arange(start, end).unsqueeze(1).expand(-1, topk).reshape(-1)
            rows.append(row.numpy())
            cols.append(idx.reshape(-1).numpy())
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        data = np.ones_like(rows, dtype=np.float32)
        return sp.csr_matrix((data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)

    def _polynomial_filter(self, matrix, coefficients):
        lam_min, lam_max = self._extreme_eigenvalues(matrix)
        lam_range = max(lam_max - lam_min, self.eps)
        shifted = matrix - lam_min * sp.eye(matrix.shape[0], format='csr', dtype=np.float32)

        result = None
        power = None
        for idx, coeff in enumerate(coefficients, start=1):
            power = shifted if idx == 1 else power.dot(shifted).tocsr()
            term = (coeff / (lam_range ** (idx - 1))) * power
            result = term if result is None else result + term
        return result.tocsr()

    def _linear_filter(self, matrix):
        lam_min, _ = self._extreme_eigenvalues(matrix)
        return (matrix - lam_min * sp.eye(matrix.shape[0], format='csr', dtype=np.float32)).tocsr()

    def _extreme_eigenvalues(self, matrix):
        n = matrix.shape[0]
        if n <= 2:
            vals = np.linalg.eigvalsh(matrix.toarray())
            return float(vals[0]), float(vals[-1])
        try:
            lam_max = sp.linalg.eigsh(matrix, k=1, which='LA', return_eigenvectors=False)[0]
            lam_min = sp.linalg.eigsh(matrix, k=1, which='SA', return_eigenvectors=False)[0]
            return float(lam_min), float(lam_max)
        except Exception:
            vals = np.linalg.eigvalsh(matrix.toarray())
            return float(vals[0]), float(vals[-1])

    def calculate_loss(self, interaction):
        return self.dummy.sum() * 0

    def full_sort_predict(self, interaction):
        users = interaction[0].detach().cpu().numpy()
        scores = self.interaction_matrix[users].dot(self.p_mm)
        if sp.issparse(scores):
            scores = scores.toarray()
        scores = np.asarray(scores, dtype=np.float32)
        return torch.from_numpy(scores).to(self.device)

    def predict(self, interaction):
        user = interaction[0]
        item = interaction[1].to(self.device)
        scores = self.full_sort_predict([user])
        return scores[torch.arange(item.shape[0], device=self.device), item]


def _safe_power(values, exponent):
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    mask = values > 0
    out[mask] = np.power(values[mask], exponent)
    return out


def _value_or_default(value, default):
    return default if value is None else value
