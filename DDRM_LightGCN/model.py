import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import pdb
import math
import time
import torch.nn.functional as F

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores= torch.sum(users_emb*pos_emb, dim=1)
        neg_scores= torch.sum(users_emb*neg_emb, dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + 
                          pos_emb.norm(2).pow(2) + 
                          neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            print('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self, diff_model, user_reverse_model, item_reverse_model, user, pos):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        recons_loss = []
        for layer in range(self.n_layers):
            # print(all_emb.shape)
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        # get batch user and item emb
        ori_user_emb = users[user]
        ori_item_emb = items[pos]

        # add noise to user and item
        noise_user_emb, noise_item_emb, ts, pt = self.apply_noise(ori_user_emb, ori_item_emb, diff_model)
        # reverse
        user_model_output = user_reverse_model(noise_user_emb, ori_item_emb, ts)
        item_model_output = item_reverse_model(noise_item_emb, ori_user_emb, ts)

        # get recons loss
        user_recons = diff_model.get_reconstruct_loss(ori_user_emb, user_model_output, pt)
        item_recons = diff_model.get_reconstruct_loss(ori_item_emb, item_model_output, pt)
        recons_loss = (user_recons + item_recons) / 2

        # update the batch user and item emb
        return user_model_output, item_model_output, recons_loss, items
    
    def computer_infer(self, user, allPos, diff_model, user_reverse_model, item_reverse_model):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]

        # get emb after GCN
        if self.config['dropout']:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph  
          
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        user_emb = users[user.long()]  
        all_aver_item_emb = []
        for pos_item in allPos:
            item_emb = items[pos_item]
            aver_item_emb = torch.mean(item_emb, dim=0)
            all_aver_item_emb.append(aver_item_emb)
        all_aver_item_emb = torch.stack(all_aver_item_emb).to(users.device)

        # # get denoised user emb
        # noise_user_emb = self.apply_T_noise(user_emb, diff_model)
        # indices = list(range(self.config['steps']))[::-1]
        # for i in indices:
        #     t = torch.tensor([i] * noise_user_emb.shape[0]).to(noise_user_emb.device)
        #     out = diff_model.p_mean_variance(user_reverse_model, noise_user_emb, all_aver_item_emb, t)
        #     if self.config['sampling_noise']:
        #         noise = torch.randn_like(noise_user_emb)
        #         nonzero_mask = (
        #             (t != 0).float().view(-1, *([1] * (len(noise_user_emb.shape) - 1)))
        #         )  # no noise when t == 0
        #         noise_user_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        #     else:
        #         noise_user_emb = out["mean"]

        noise_user_emb = user_emb

        # get generated item          
        # reverse
        noise_emb = self.apply_T_noise(all_aver_item_emb, diff_model)
        indices = list(range(self.config['sampling_steps']))[::-1]
        for i in indices:
            t = torch.tensor([i] * noise_emb.shape[0]).to(noise_emb.device)
            out = diff_model.p_mean_variance(item_reverse_model, noise_emb, noise_user_emb, t)
            if self.config['sampling_noise']:
                noise = torch.randn_like(noise_emb)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(noise_emb.shape) - 1)))
                )  # no noise when t == 0
                noise_emb = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
            else:
                noise_emb = out["mean"]

        return noise_emb, items
    
    def apply_noise(self, user_emb, item_emb, diff_model):
        # cat_emb shape: (batch_size*3, emb_size)
        emb_size = user_emb.shape[0]
        ts, pt = diff_model.sample_timesteps(emb_size, 'uniform')
        # ts_ = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)

        # add noise to users
        user_noise = torch.randn_like(user_emb)
        item_noise = torch.randn_like(item_emb)
        user_noise_emb = diff_model.q_sample(user_emb, ts, user_noise)
        item_noise_emb = diff_model.q_sample(item_emb, ts, item_noise)
        return user_noise_emb, item_noise_emb, ts, pt

    def apply_noise_sample(self, cat_emb, diff_model):
        t = torch.tensor([self.config['sampling_steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)
        noise = torch.randn_like(cat_emb)
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb

    def apply_T_noise(self, cat_emb, diff_model):
        t = torch.tensor([self.config['steps'] - 1] * cat_emb.shape[0]).to(cat_emb.device)
        noise = torch.randn_like(cat_emb)
        noise_emb = diff_model.q_sample(cat_emb, t, noise)
        return noise_emb
    
    def getUsersRating(self, users, train_dict, user_reverse_model, item_reverse_model, diff_model):
        item_emb, all_items = self.computer_infer(users, train_dict, diff_model, user_reverse_model, item_reverse_model)
        rating = self.rounding_inner(item_emb, all_items)
        return rating
    
    def rounding_inner(self, item_emb, all_items):
        item_emb_expanded = item_emb.unsqueeze(1)  # Shape: [bs_user, 1, emb]
        all_items_expanded = all_items.unsqueeze(0)  # Shape: [1, item_num, emb]

        # Element-wise multiplication
        dot_product = torch.sum(item_emb_expanded * all_items_expanded, dim=2) 

        return dot_product
    
    def rounding_cos(self, item_emb, all_items):
        item_emb_normalized = F.normalize(item_emb, p=2, dim=1)  # Shape: [bs_user, emb]
        all_items_normalized = F.normalize(all_items, p=2, dim=1)  # Shape: [item_num, emb]

        # Calculate cosine similarity
        cos_sim_matrix = torch.mm(item_emb_normalized, all_items_normalized.t())
        
        return cos_sim_matrix

    def rounding_mse(self, item_emb, all_items):
        # item_emb shape: [bs_user, emb]
        # all_item shape: [item_num, emb]
        squared_diffs = (item_emb.unsqueeze(1) - all_items.unsqueeze(0)) ** 2
        mse = torch.mean(squared_diffs, dim=-1)
        return mse
    
    def getEmbedding(self, users, pos_items, neg_items, user_reverse_model, item_reverse_model, diff_model):
        users_emb, pos_emb, recons_loss, all_items = self.computer(diff_model, user_reverse_model, item_reverse_model, users, pos_items)
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego, recons_loss
    
    def bpr_loss(self, users, pos, neg, user_reverse_model, item_reverse_model, diff_model):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0, reconstruct_loss) = self.getEmbedding(users.long(), pos.long(), neg.long(), user_reverse_model, item_reverse_model, diff_model)


        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.nn.functional.softplus(neg_scores - pos_scores)
        
        return loss, reg_loss, reconstruct_loss, pos_scores
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class Diff_Transformer(nn.Module):
    def __init__(self, in_dims, out_dims, w_in_dims, w_out_dims, norm=False, dropout=0.5):
        super(Diff_Transformer, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.w_in_dims = w_in_dims
        self.w_out_dims = w_out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.norm = norm

        in_dims_temp = self.in_dims
        out_dims_temp = self.out_dims

        w_in_dims_temp = self.w_in_dims
        w_out_dims_temp = self.w_out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.wk_in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_in_dims_temp[:-1], w_in_dims_temp[1:])])
        self.wk_out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_out_dims_temp[:-1], w_out_dims_temp[1:])])
        
        self.wv_in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_in_dims_temp[:-1], w_in_dims_temp[1:])])
        self.wv_out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_out_dims_temp[:-1], w_out_dims_temp[1:])])
        
        self.wq_in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_in_dims_temp[:-1], w_in_dims_temp[1:])])
        self.wq_out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(w_out_dims_temp[:-1], w_out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        

        for layer in self.wk_in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.wk_out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.wv_in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.wv_out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.wq_in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.wq_out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        
        # size = self.emb_layer.weight.size()
        # fan_out = size[0]
        # fan_in = size[1]
        # std = np.sqrt(2.0 / (fan_in + fan_out))
        # self.emb_layer.weight.data.normal_(0.0, std)
        # self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, noise_emb, con_emb, timesteps):
        # noise emb: [bs, emb]
        time_emb = timestep_embedding(timesteps, world.config['latent_dim_rec']).to(noise_emb.device)
        kv_emb = time_emb + noise_emb
        k_emb = kv_emb
        v_emb = kv_emb
        q_emb = con_emb

        for i, layer in enumerate(self.wk_in_layers):
            k_emb = layer(k_emb)
            k_emb = torch.tanh(k_emb)
        for i, layer in enumerate(self.wk_out_layers):
            k_emb = layer(k_emb)
            if i != len(self.wk_out_layers) - 1:
                k_emb = torch.tanh(k_emb)

        for i, layer in enumerate(self.wv_in_layers):
            v_emb = layer(v_emb)
            v_emb = torch.tanh(v_emb)
        for i, layer in enumerate(self.wv_out_layers):
            v_emb = layer(v_emb)
            if i != len(self.wv_out_layers) - 1:
                v_emb = torch.tanh(v_emb)

        for i, layer in enumerate(self.wq_in_layers):
            q_emb = layer(q_emb)
            q_emb = torch.tanh(q_emb)
        for i, layer in enumerate(self.wq_out_layers):
            q_emb = layer(q_emb)
            if i != len(self.wq_out_layers) - 1:
                q_emb = torch.tanh(q_emb)

        sim_emb = k_emb * q_emb
        all_emb = torch.sigmoid(sim_emb) * v_emb
        # all_emb = sim_emb + v_emb

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            all_emb = torch.tanh(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                all_emb = torch.tanh(all_emb)
        return all_emb

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0]*2 + self.time_emb_dim] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)
        out_dims_temp = self.out_dims
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    # def forward(self, x, timesteps):
    #     time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
    #     emb = self.emb_layer(time_emb)
    #     if self.norm:
    #         x = F.normalize(x)
    #     x = self.drop(x)

    #     h = torch.cat([x, emb], dim=-1)

    #     for i, layer in enumerate(self.in_layers):
    #         h = layer(h)
    #         h = torch.tanh(h)
    #     for i, layer in enumerate(self.out_layers):
    #         h = layer(h)
    #         if i != len(self.out_layers) - 1:
    #             h = torch.tanh(h)
    #     return h

    def forward(self, noise_emb, con_emb, timesteps):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(noise_emb.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            noise_emb = F.normalize(noise_emb)
        noise_emb = self.drop(noise_emb)

        all_emb = torch.cat([noise_emb, emb, con_emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            all_emb = layer(all_emb)
            if world.config['act'] == 'tanh':
                all_emb = torch.tanh(all_emb)
            elif world.config['act'] == 'sigmoid':
                all_emb = torch.sigmoid(all_emb)
            elif world.config['act'] == 'relu':
                all_emb = F.relu(all_emb)
        for i, layer in enumerate(self.out_layers):
            all_emb = layer(all_emb)
            if i != len(self.out_layers) - 1:
                if world.config['act'] == 'tanh':
                    all_emb = torch.tanh(all_emb)
                elif world.config['act'] == 'sigmoid':
                    all_emb = torch.sigmoid(all_emb)
                elif world.config['act'] == 'relu':
                    all_emb = F.relu(all_emb)
        return all_emb

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

