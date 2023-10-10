import torch
from torch import nn
from Params import args
from model.GAT import SpGAT
from dataloader import MBHSRecDataset
from Params import args
import os
import numpy as np
import torch.nn.functional as F


class MBSoc(nn.Module):
    def __init__(self, num_users, num_items, train_set, device, pre_train=True):
        super(MBSoc, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.device = device
        self.pretrain = pre_train
        self.emb_dim = args.embed_dim
        self.behav_num = args.behav_num
        self.path = 'dataset/' + args.dataset
        self.init_model()
        self.dataset = MBHSRecDataset(num_users, num_items, train_set, device)
        self.graph = self.dataset.graph
        self.behav_graph_list = self.dataset.behav_graph_list
        self.social_graph = self.dataset.social_graph
        self.social_aggregator = SpGAT(self.emb_dim, self.emb_dim, self.emb_dim, args.drop_rate, alpha=args.alpha, nheads=1)
        self.tau = args.tau
        self.n_layers = args.layers
        self.rate_pred = nn.Sequential(
            nn.Linear(2*self.emb_dim, 1),
        )
    
    def init_model(self):
        self.embedding_user = nn.Embedding(self.num_users, self.emb_dim)
        self.embedding_item = nn.Embedding(self.num_items, self.emb_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        print('MBSoc: Use NORMAL distribution initilizer for User and Item')     
        self.embedding_dict=nn.ParameterDict({
            'embedding_user_behav_0': nn.Parameter(nn.init.normal_(torch.empty(self.num_users, self.emb_dim))),
            'embedding_item_behav_0': nn.Parameter(nn.init.normal_(torch.empty(self.num_items, self.emb_dim))),
            'embedding_user_behav_1': nn.Parameter(nn.init.normal_(torch.empty(self.num_users, self.emb_dim))),
            'embedding_item_behav_1': nn.Parameter(nn.init.normal_(torch.empty(self.num_items, self.emb_dim))),
            'embedding_user_behav_2': nn.Parameter(nn.init.normal_(torch.empty(self.num_users, self.emb_dim))),
            'embedding_item_behav_2': nn.Parameter(nn.init.normal_(torch.empty(self.num_items, self.emb_dim))),
        })
        if not self.pretrain:
            try:
                for b in range(self.behav_num):
                    pretrain_user_embedding = np.load(self.path + '/embedding/pretrain_user_behav_'+ str(b) +'.npz', allow_pickle=True)['save_user']
                    pretrain_item_embedding = np.load(self.path + '/embedding/pretrain_item_behav_'+ str(b) +'.npz', allow_pickle=True)['save_item']
                    self.embedding_dict['embedding_user_behav_'+str(b)].data.copy_(torch.from_numpy(pretrain_user_embedding))
                    self.embedding_dict['embedding_item_behav_'+str(b)].data.copy_(torch.from_numpy(pretrain_item_embedding))
                    self.embedding_dict['embedding_user_behav_'+str(b)].requires_grad = True
                    self.embedding_dict['embedding_item_behav_'+str(b)].requires_grad = True
                print("MBSoc: Load Pre-Trained Single behavior User-Item Embedding")
            except:
                print('MBSoc: Loading Failed--Use NORMAL distribution initilizer')

    def InfoNCE(self, view1, view2, temperature, b_cos=True):
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score+10e-6)
        return torch.mean(cl_loss)
    
    def cal_loss(self, ids, labels, behav_graph_list):
        uids, iids = ids[0], ids[1]
        # Calculate main loss
        view_list, main_loss_list, cl_sim_list, cl_diff_list = [], [], [], []
        user_view_main, item_view_main = self.forward()
        outputs = self.rate_pred(torch.cat([user_view_main[uids.long()], item_view_main[iids.long()]], dim=1))
        main_loss_list.append(nn.MSELoss()(outputs, labels.unsqueeze(1)))
        l2_loss = self.l2_reg_loss(args.reg_rate, user_view_main[uids.long()], item_view_main[iids.long()])
        view_main = torch.cat([user_view_main[uids.long()], item_view_main[iids.long()]], dim=0)
       
        # Calculate cl loss       
        for b in range(self.behav_num):
            user_view_b, item_view_b = self.forward(behav_graph_list[b], b)
            outputs = self.rate_pred(torch.cat([user_view_b[uids.long()], item_view_b[iids.long()]], dim=1))
            main_loss_list.append(nn.MSELoss()(outputs, labels.unsqueeze(1)))
            l2_loss += self.l2_reg_loss(args.reg_rate, user_view_b[uids.long()], item_view_b[iids.long()])
            view = torch.cat([user_view_b[uids.long()], item_view_b[iids.long()]], dim=0)
            view_list.append(view)

        for view in view_list:
            cl_sim_list.append(self.InfoNCE(view_main, view, self.tau))
            for view_diff in view_list:
                if not view_diff.equal(view):
                    cl_diff_list.append(self.InfoNCE(view, view_diff, self.tau))
                    
        cl_sim = torch.mean(torch.stack(cl_sim_list, dim=0))
        cl_diff = torch.mean(torch.stack(cl_diff_list, dim=0))
        main_loss = torch.mean(torch.stack(main_loss_list, dim=0))
        cl_loss = args.behav_loss_rate* (cl_sim - cl_diff)

        return main_loss, cl_loss, l2_loss
    
    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2)/emb.shape[0]
        return emb_loss * reg

    def graph_reconstruction(self, behav_index=None):
        '''
        Generate the single behavior graph
        '''
        graph_adjs = []
        for _ in range(self.n_layers):
            if behav_index in range(self.behav_num):
                graph_adjs.append(self.behav_graph_list[behav_index])
            else:
                graph_adjs.append(self.graph)
        return graph_adjs

    def getUserRating(self, users, items, behavior_adj=None, behav_index=None):
        '''
        Output the prediction 
        '''
        all_users, all_items = self.forward(behavior_adj, behav_index)
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        rating = self.rate_pred(torch.cat([users_emb, items_emb], dim=1))
        return rating

    def forward(self, behavior_adj=None, behav_index=None):
        if behav_index in range(self.behav_num):
            users_emb = self.embedding_dict['embedding_user_behav_'+str(behav_index)]
            items_emb = self.embedding_dict['embedding_item_behav_'+str(behav_index)]
        else:
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
        ego_embeddings = torch.cat([users_emb, items_emb])
        all_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            if behavior_adj is not None:
                if isinstance(behavior_adj, list):
                    ego_embeddings = torch.sparse.mm(behavior_adj[k], ego_embeddings)
                else:
                    ego_embeddings = torch.sparse.mm(behavior_adj, ego_embeddings)
            else:
                ego_embeddings = torch.sparse.mm(self.graph, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        social_embeddings = self.social_aggregator(user_all_embeddings, self.social_graph)
        user_all_embeddings = (user_all_embeddings + social_embeddings) / 2
        return user_all_embeddings, item_all_embeddings

    def save_embedding(self, behav_index=None):
        if self.pretrain:
            save_path = self.path + '/embedding'
            if not os.path.exists(save_path):
                 os.makedirs(save_path)
            if behav_index is None:
                np.savez(save_path + '/pretrain_user.npz', save_user = self.embedding_user.weight.data.cpu().numpy())
                np.savez(save_path + '/pretrain_item.npz', save_item = self.embedding_item.weight.data.cpu().numpy())
            # print("Successfully save user-item embedding")
            else:
                np.savez(save_path + '/pretrain_user_behav_'+ str(behav_index) +'.npz', save_user = self.embedding_dict['embedding_user_behav_'+str(behav_index)].data.cpu().numpy())
                np.savez(save_path + '/pretrain_item_behav_'+ str(behav_index) +'.npz', save_item = self.embedding_dict['embedding_item_behav_'+str(behav_index)].data.cpu().numpy())

