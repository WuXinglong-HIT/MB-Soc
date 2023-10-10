import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from Params import args
from random import shuffle
import os
import pickle
import numpy as np
from model.MBSoc import MBSoc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def pre_train(user_count, item_count, train_set, valid_set, behavior=None):
    print("\n", args, "\n")
    model = MBSoc(user_count, item_count, train_set, device, pre_train=True).to(device)
    pre_bestMAE = trainAndTest(train_set, valid_set, model, behavior)
    print("Best MAE of Pre-training Model: {}".format(pre_bestMAE))

def trainAndTest(train_set, valid_set, model, behavior=None):
    shuffle(train_set)
    optimizer = optim.Adam(model.parameters(), args.lr)

    best_MAE, num_fail = 10e3, 1
    summaryWriter = SummaryWriter("./runs/MBSoc")
    behav_adj_list = [model.graph_reconstruction(behav_index=b) for b in range(args.behav_num)]

    model_path = 'checkpoints/' + args.model_name + '/' + args.dataset + '_' + str(1-2*args.test_rate)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for ep in range(args.pre_epoch):
        checkpoint_path = model_path + "/pretrain-best.pt"
        if ep % 2 == 0:
            MAE, RMSE = test_MBSoc(valid_set, model, behav_adj_list, behavior)
            if best_MAE > MAE:
                best_MAE = MAE
                model.save_embedding(behavior)
                torch.save(model, checkpoint_path)
                num_fail = 1
            elif num_fail < args.early_steps:
                num_fail += 1
            else:
                break
            summaryWriter.add_scalar("MAE", MAE, ep+1)
            summaryWriter.add_scalar("RMSE", RMSE, ep+1)
            print("Epoch {} Validate: MAE: {:.4f}, RMSE: {:.4f}, best MAE: {:.4f}".format(ep+1, MAE, RMSE, best_MAE))
        output_information = train_model(train_set, model, optimizer, behav_adj_list)
        print(f'Epoch[{ep+1}/{args.pre_epoch}] {output_information}')

    summaryWriter.close()
    return best_MAE

def train_model(dataset, model, optimizer, behav_adj_list=None):
    total_loss = 0
    for _, batch in enumerate(next_batch_pairwise(dataset, args.batch)):
        user_idx, pos_idx, neg_idx, labels = batch
        user_idx = torch.tensor(user_idx, device=device)
        pos_idx = torch.tensor(pos_idx, device=device)
        neg_idx = torch.tensor(neg_idx, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        _, cl_loss, _ = model.cal_loss([user_idx.long(), pos_idx.long()], labels, behav_adj_list)
        batch_loss = cl_loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return f"loss:{cl_loss:.3f}"

def test_MBSoc(dataset, model, behav_adj_list=None, behavior=None):
    '''
    Test the model and output the matrices [MAE, RMSE]
    '''
    model.eval()
    test_loss = []
    with torch.no_grad(): 
        for _, batch in enumerate(next_test_batch(dataset, args.batch)):
            users, items, labels = batch
            users = torch.tensor(users, device=device)
            items = torch.tensor(items, device=device)
            labels = torch.tensor(labels, device=device)
            if behavior is None:
                preds = model.getUserRating(users, items)
            else:
                preds = model.getUserRating(users, items, behav_adj_list[behavior], behavior)
            loss = torch.abs(preds.squeeze(1) - labels)
            test_loss.extend(loss.data.cpu().numpy().tolist())
        MAE = np.mean(test_loss)
        RMSE = np.sqrt(np.mean(np.power(test_loss, 2)))

    return MAE, RMSE

def next_batch_pairwise(train_data, batch_size):
    '''
    Generate training data 
    '''
    ptr = 0
    data_size = len(train_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
            break
        users = np.array(train_data)[ptr:batch_end, 0]
        items = np.array(train_data)[ptr:batch_end, 1]
        labels = np.array(train_data)[ptr:batch_end, 2]
        users = np.array(users).astype(int)
        ptr = batch_end

        # Randomly generate a negative sample
        with open('dataset/' + args.dataset + '/MBSoc/pos_items.pkl', 'rb') as f:
            pos_items = pickle.load(f)
        neg_items = []
        user_num  = len(pos_items)
        for u in users:
            pos_users = pos_items[u]
            neg_users = set(range(user_num))-set(pos_users)
            neg_user = np.random.choice(list(neg_users))
            neg_items.append(neg_user)

        yield users, items, neg_items, labels

def next_test_batch(test_data, batch_size):
    '''
    Generate testing data 
    '''
    ptr = 0
    data_size = len(test_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
            break
        users = np.array(test_data)[ptr:batch_end, 0]
        items =  np.array(test_data)[ptr:batch_end, 1]
        labels =  np.array(test_data)[ptr:batch_end, 2]
        ptr = batch_end

        yield users, items, labels


if __name__ == '__main__':
    with open('dataset/' + args.dataset + '/MBSoc/all_data.pkl', 'rb') as f:
        (user_count, item_count, rating_count) = pickle.load(f)
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open('dataset/' + args.dataset + '/MBSoc/behavior_train_set.pkl', 'rb') as f:
        train_set_b0 = pickle.load(f)
        train_set_b1 = pickle.load(f)
        train_set_b2 = pickle.load(f)

    train_list = [[0, train_set_b0], [1, train_set_b1], [2, train_set_b2]]
    for b, train_set in train_list:
        pre_train(user_count, item_count, train_set, valid_set, b)
 
    