import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse as sp
from Params import args
import pickle
import os
    

class MBHSRecDataset(Dataset):
    def __init__(self, user_num, item_num, trainset, device='cuda'):
        self.user_num = user_num
        self.item_num = item_num
        self.trainset = trainset
        self.device = device
        self.__generate_set()
 
    def __generate_set(self):
        self.user_list = np.array(self.trainset)[:, 0]
        self.item_list = np.array(self.trainset)[:, 1]
        self.rating_list = np.array(self.trainset)[:, 2]

    def generate_multi_behav_set(self, data):
        user_list = np.array(data)[:, 0]
        item_list = np.array(data)[:, 1]
        rating_list = np.array(data)[:, 2]
        return user_list, item_list, rating_list

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def generate_sym_degree_matrix(self, user_count, item_count, device='cuda'):
        path = 'dataset/' + args.dataset + '/MBSoc'
        try:
            pre_adj_mat = sp.load_npz(path + '/s_pre_adj_mat.npz')
            print("Successfully load adjacency matrix")
            norm_adj = pre_adj_mat
        except:
            uid_list, iid_list = self.user_list, self.item_list
            assert len(uid_list) == len(iid_list)
            # Generate the user-item interaction bipartite graph
            data = np.ones(len(uid_list))
            coo_matrix = sp.coo_matrix((data, (uid_list, iid_list)), shape=(user_count, item_count))
            R = coo_matrix.tolil()

            # Calculate D(^-1/2)AD(^-1/2)
            degree_matrix = sp.dok_matrix((user_count+item_count, user_count+item_count), dtype=np.float32)
            degree_matrix = degree_matrix.tolil()
            degree_matrix[:user_count, user_count:] = R
            degree_matrix[user_count:, :user_count] = R.T
            degree_matrix = degree_matrix.todok()

            rowsum = np.array(degree_matrix.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(degree_matrix)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()

            sp.save_npz(path + '/s_pre_adj_mat.npz', norm_adj)
            print("Successfully generate and save adjacency matrix")
        sym_degree_matrix = self.convert_sp_mat_to_sp_tensor(norm_adj)
        sym_degree_matrix = sym_degree_matrix.coalesce().to(device)

        return sym_degree_matrix

    def generate_mul_behav_adj_matrix(self, user_count, item_count, device='cuda'):
        path = 'dataset/' + args.dataset + '/MBSoc'
        try:
            behav_graph_list = []
            for b in range(args.behav_num):
                norm_adj_b = sp.load_npz(path + '/behav_graph_list/behavior_'+ str(b) +'.npz',)
                behav_graph_list.append(self.convert_sp_mat_to_sp_tensor(norm_adj_b).coalesce().to(device))
            print("Successfully load multiple behaviors' adjacency matrix...")
        except:
            # Load the user-item interaction bipartite graph by behavior
            with open(path + '/behavior_train_set.pkl', 'rb') as f:
                train_set_b0 = pickle.load(f)
                train_set_b1 = pickle.load(f)
                train_set_b2 = pickle.load(f)
            data_list = [train_set_b0, train_set_b1, train_set_b2]
            behav_graph_list = []
            for data_b in data_list:
                uid_list, iid_list, rating_list = self.generate_multi_behav_set(data_b)
                assert len(uid_list) == len(iid_list) and len(uid_list) == len(rating_list) 
                rating_list = np.ones_like(rating_list)           
                rating_matrix = sp.coo_matrix((rating_list, (uid_list, iid_list)), shape=(user_count, item_count))
                R = rating_matrix.tolil()

                degree_matrix = sp.dok_matrix((user_count+item_count, user_count+item_count), dtype=np.float32)
                degree_matrix = degree_matrix.tolil()
                degree_matrix[:user_count, user_count:] = R
                degree_matrix[user_count:, :user_count] = R.T
                degree_matrix = degree_matrix.todok()

                rowsum = np.array(degree_matrix.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                norm_adj = d_mat.dot(degree_matrix)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()

                behav_graph_list.append(self.convert_sp_mat_to_sp_tensor(norm_adj).coalesce().to(device))
                if not os.path.exists(path + '/behav_graph_list'):
                    os.makedirs(path + '/behav_graph_list')
                sp.save_npz(path + '/behav_graph_list/behavior_'+ str(b) +'.npz', norm_adj)
                print("Successfully generate and save multiple behaviors' adjacency matrix...")

        return  behav_graph_list 

    def generate_social_graph(self, device='cuda'):
        with open('dataset/' + args.dataset + '/MBSoc/social_relation.pkl', 'rb') as f:
            source_list = pickle.load(f)
            target_list = pickle.load(f)
        trust_list = np.ones_like(source_list)
        trust_matrix = sp.coo_matrix((trust_list, (source_list, target_list)), shape=(self.user_num, self.user_num))
        trust_matrix = self.convert_sp_mat_to_sp_tensor(trust_matrix).coalesce().to(device)
        return trust_matrix
    
    @property
    def graph(self):
        return self.generate_sym_degree_matrix(self.user_num, self.item_num, device=self.device)

    @property
    def behav_graph_list(self):
        behav_graph_list = self.generate_mul_behav_adj_matrix(self.user_num, self.item_num, device=self.device)
        return behav_graph_list
    
    @property
    def training_data(self):
        return self.trainset

    @property
    def social_graph(self):
        social_graph = self.generate_social_graph(device=self.device)
        return social_graph

