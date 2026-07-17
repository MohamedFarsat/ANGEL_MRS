# coding: utf-8
#
# Ported from the official MENTOR implementation:
# https://github.com/Jinfeng-Xu/MENTOR

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender


class MENTOR(GeneralRecommender):
    def __init__(self, config, dataset):
        super(MENTOR, self).__init__(config, dataset)

        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.knn_block_size = config['knn_block_size'] or 1024
        self.mm_image_weight = config['mm_image_weight']
        self.batch_size = config['train_batch_size']
        self.num_user = self.n_users
        self.num_item = self.n_items
        self.k = 40
        self.aggr_mode = 'add'
        self.dataset = dataset
        self.dropout = config['dropout']
        self.reg_weight = config['reg_weight']
        self.align_weight = config['align_weight']
        self.mask_weight_g = config['mask_weight_g']
        self.mask_weight_f = config['mask_weight_f']
        self.temp = config['temp']
        self.drop_rate = 0.1
        self.v_rep = None
        self.t_rep = None
        self.v_preference = None
        self.t_preference = None
        self.id_preference = None
        self.dim_latent = 64
        self.mm_adj = None
        self.mlp = nn.Linear(2 * dim_x, 2 * dim_x)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(
            os.path.join(dataset_path, config['user_graph_dict_file']),
            allow_pickle=True
        ).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file, map_location=self.device)
        else:
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
            torch.save(self.mm_adj, mm_adj_file)

        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 2, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.item_index = torch.arange(self.num_item, dtype=torch.long)
        drop_item = torch.tensor(
            np.random.choice(self.item_index, int(self.num_item * self.drop_rate), replace=False))
        drop_item_single = drop_item[:len(drop_item)]
        self.dropv_node_idx = drop_item_single[:int(len(drop_item_single) / 3)]
        self.dropt_node_idx = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv, mask_dropt = [], []
        for idx, num in enumerate(mask_cnt):
            mask_dropv.extend([False] * num if idx in self.dropv_node_idx else [True] * num)
            mask_dropt.extend([False] * num if idx in self.dropt_node_idx else [True] * num)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        self.edge_index_dropv = torch.tensor(edge_index[mask_dropv]).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index[mask_dropt]).t().contiguous().to(self.device)
        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)

        if self.v_feat is not None:
            self.v_gcn = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                             dim_latent=64, device=self.device, features=self.v_feat)
            self.v_gcn_n1 = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                                dim_latent=64, device=self.device, features=self.v_feat)
            self.v_gcn_n2 = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                                dim_latent=64, device=self.device, features=self.v_feat)
        if self.t_feat is not None:
            self.t_gcn = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                             dim_latent=64, device=self.device, features=self.t_feat)
            self.t_gcn_n1 = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                                dim_latent=64, device=self.device, features=self.t_feat)
            self.t_gcn_n2 = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                                dim_latent=64, device=self.device, features=self.t_feat)

        self.id_feat = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.n_items, self.dim_latent), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))
        self.id_gcn = GCN(dataset, self.batch_size, self.num_user, self.num_item, dim_x, self.aggr_mode,
                          dim_latent=64, device=self.device, features=self.id_feat)

        init_embed = nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user + self.num_item, dim_x), dtype=torch.float32)
        ).to(self.device)
        self.register_buffer('result_embed', init_embed)
        self.register_buffer('result_embed_guide', torch.empty_like(init_embed))
        self.register_buffer('result_embed_v', torch.empty_like(init_embed))
        self.register_buffer('result_embed_t', torch.empty_like(init_embed))
        self.register_buffer('result_embed_n1', torch.empty_like(init_embed))
        self.register_buffer('result_embed_n2', torch.empty_like(init_embed))

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True).clamp_min(1e-12))
        n_items = context_norm.shape[0]
        topk = min(self.knn_k, n_items)
        row_chunks, col_chunks = [], []
        for start in range(0, n_items, self.knn_block_size):
            end = min(start + self.knn_block_size, n_items)
            sim = torch.mm(context_norm[start:end], context_norm.transpose(1, 0))
            _, knn_ind = torch.topk(sim, topk, dim=-1)
            rows = torch.arange(start, end, device=self.device).unsqueeze(1).expand(-1, topk)
            row_chunks.append(rows.reshape(-1))
            col_chunks.append(knn_ind.reshape(-1))
        indices = torch.stack((torch.cat(row_chunks), torch.cat(col_chunks)), 0)
        adj_size = torch.Size((n_items, n_items))
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size).to(self.device)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        values = r_inv_sqrt[indices[0]] * r_inv_sqrt[indices[1]]
        return torch.sparse.FloatTensor(indices, values, adj_size).to(self.device)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def InfoNCE(self, view1, view2, temp):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = torch.exp((view1 * view2).sum(dim=-1) / temp)
        ttl_score = torch.exp(torch.matmul(view1, view2.transpose(0, 1)) / temp).sum(dim=1)
        return torch.mean(-torch.log(pos_score / ttl_score.clamp_min(1e-12)))

    def forward(self, interaction):
        user_nodes = interaction[0]
        pos_item_nodes = interaction[1] + self.n_users
        neg_item_nodes = interaction[2] + self.n_users

        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv, self.edge_index, self.v_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt, self.edge_index, self.t_feat)
        self.id_rep, self.id_preference = self.id_gcn(self.edge_index_dropt, self.edge_index, self.id_feat)
        self.v_rep_n1, _ = self.v_gcn_n1(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n1, _ = self.t_gcn_n1(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)
        self.v_rep_n2, _ = self.v_gcn_n2(self.edge_index_dropv, self.edge_index, self.v_feat, perturbed=True)
        self.t_rep_n2, _ = self.t_gcn_n2(self.edge_index_dropt, self.edge_index, self.t_feat, perturbed=True)

        representation = torch.cat((self.v_rep, self.t_rep), dim=1)
        guide_representation = torch.cat((self.id_rep, self.id_rep), dim=1)
        v_representation = torch.cat((self.v_rep, self.v_rep), dim=1)
        t_representation = torch.cat((self.t_rep, self.t_rep), dim=1)
        representation_n1 = torch.cat((self.v_rep_n1, self.t_rep_n1), dim=1)
        representation_n2 = torch.cat((self.v_rep_n2, self.t_rep_n2), dim=1)

        user_rep = self._weighted_user_rep(self.v_rep, self.t_rep)
        guide_user_rep = torch.cat((self.id_rep[:self.num_user], self.id_rep[:self.num_user]), dim=1)
        v_user_rep = torch.cat((self.v_rep[:self.num_user], self.v_rep[:self.num_user]), dim=1)
        t_user_rep = torch.cat((self.t_rep[:self.num_user], self.t_rep[:self.num_user]), dim=1)
        user_rep_n1 = self._weighted_user_rep(self.v_rep_n1, self.t_rep_n1)
        user_rep_n2 = self._weighted_user_rep(self.v_rep_n2, self.t_rep_n2)

        item_rep = representation[self.num_user:] + self.buildItemGraph(representation[self.num_user:])
        guide_item_rep = guide_representation[self.num_user:] + self.buildItemGraph(guide_representation[self.num_user:])
        v_item_rep = v_representation[self.num_user:] + self.buildItemGraph(v_representation[self.num_user:])
        t_item_rep = t_representation[self.num_user:] + self.buildItemGraph(t_representation[self.num_user:])
        item_rep_n1 = representation_n1[self.num_user:] + self.buildItemGraph(representation_n1[self.num_user:])
        item_rep_n2 = representation_n2[self.num_user:] + self.buildItemGraph(representation_n2[self.num_user:])

        self.user_rep, self.item_rep = user_rep, item_rep
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        self.result_embed_guide = torch.cat((guide_user_rep, guide_item_rep), dim=0)
        self.result_embed_v = torch.cat((v_user_rep, v_item_rep), dim=0)
        self.result_embed_t = torch.cat((t_user_rep, t_item_rep), dim=0)
        self.result_embed_n1 = torch.cat((user_rep_n1, item_rep_n1), dim=0)
        self.result_embed_n2 = torch.cat((user_rep_n2, item_rep_n2), dim=0)

        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        return torch.sum(user_tensor * pos_item_tensor, dim=1), torch.sum(user_tensor * neg_item_tensor, dim=1)

    def _weighted_user_rep(self, v_rep, t_rep):
        user_rep = torch.stack((v_rep[:self.num_user], t_rep[:self.num_user]), dim=2)
        user_rep = self.weight_u.transpose(1, 2) * user_rep
        return torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

    def buildItemGraph(self, h):
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        return h

    def fit_Gaussian_dis(self):
        return (
            torch.var(self.result_embed), torch.mean(self.result_embed),
            torch.var(self.result_embed_guide), torch.mean(self.result_embed_guide),
            torch.var(self.result_embed_v), torch.mean(self.result_embed_v),
            torch.var(self.result_embed_t), torch.mean(self.result_embed_t),
        )

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores).clamp_min(1e-12)))

        reg_embedding_loss_v = (self.v_preference[user] ** 2).mean() if self.v_preference is not None else 0.0
        reg_embedding_loss_t = (self.t_preference[user] ** 2).mean() if self.t_preference is not None else 0.0
        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()

        with torch.no_grad():
            u_temp, i_temp = self.user_rep.clone(), self.item_rep.clone()
            u_temp2, i_temp2 = self.mlp(self.user_rep.clone()), self.mlp(self.item_rep.clone())
            u_temp = F.dropout(u_temp, self.dropout)
            i_temp = F.dropout(i_temp, self.dropout)
        mask_loss_u = 1 - F.cosine_similarity(u_temp, u_temp2).mean()
        mask_loss_i = 1 - F.cosine_similarity(i_temp, i_temp2).mean()
        mask_f_loss = self.mask_weight_f * (mask_loss_i + mask_loss_u)

        r_var, r_mean, g_var, g_mean, v_var, v_mean, t_var, t_mean = self.fit_Gaussian_dis()
        align_loss = (
            (torch.abs(g_var - r_var) + torch.abs(g_mean - r_mean)).mean() +
            (torch.abs(g_var - v_var) + torch.abs(g_mean - v_mean)).mean() +
            (torch.abs(g_var - t_var) + torch.abs(g_mean - t_mean)).mean() +
            (torch.abs(r_var - v_var) + torch.abs(r_mean - v_mean)).mean() +
            (torch.abs(r_var - t_var) + torch.abs(r_mean - t_mean)).mean() +
            (torch.abs(v_var - t_var) + torch.abs(v_mean - t_mean)).mean()
        ) * self.align_weight

        mask_g_loss = (
            self.InfoNCE(self.result_embed_n1[:self.n_users], self.result_embed_n2[:self.n_users], self.temp) +
            self.InfoNCE(self.result_embed_n1[self.n_users:], self.result_embed_n2[self.n_users:], self.temp)
        ) * self.mask_weight_g

        return loss_value + reg_loss + align_loss + mask_f_loss + mask_g_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]
        return torch.matmul(user_tensor[interaction[0], :], item_tensor.t())

    def topk_sample(self, k):
        user_graph_index = []
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        fallback = [0] * k
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) == 0:
                user_graph_index.append(fallback)
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]
            while len(user_graph_sample) < k:
                rand_index = np.random.randint(0, len(user_graph_sample))
                user_graph_sample.append(user_graph_sample[rand_index])
                user_graph_weight.append(user_graph_weight[rand_index])
            user_graph_index.append(user_graph_sample)
            user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)
        return user_graph_index, user_weight_matrix


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode,
                 dim_latent=None, device=None, features=None):
        super(GCN, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_latent), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
        else:
            self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
                np.random.randn(num_user, self.dim_feat), dtype=torch.float32, requires_grad=True),
                gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)

    def forward(self, edge_index_drop, edge_index, features, perturbed=False):
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        h = self.conv_embed_1(x, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h)
            h = h + torch.sign(h) * F.normalize(random_noise, dim=-1) * 0.1
        h_1 = self.conv_embed_1(h, edge_index)
        if perturbed:
            random_noise = torch.rand_like(h_1)
            h_1 = h_1 + torch.sign(h_1) * F.normalize(random_noise, dim=-1) * 0.1
        return x + h + h_1, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out
