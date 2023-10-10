import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
import os
import pickle
import logging
from Params import args
from random import shuffle
from model.MBSoc import MBSoc
from pre_training import test_MBSoc, next_batch_pairwise


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.DEBUG,
                    filename='./Log/'+args.dataset+'_Train.log',
                    filemode='a',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'#
                    )

def fine_tune(user_count, item_count, train_set, valid_set):
    logging.info(args)
    model_ft = MBSoc(user_count, item_count, train_set, device, pre_train=False).to(device)
    pre_bestMAE_ft = trainAndTest(train_set, valid_set, model_ft)
    print("Best MAE of Fine-Tuning Model: {}".format(pre_bestMAE_ft))

def trainAndTest(train_set, valid_set, model):
    shuffle(train_set)
    optimizer = optim.AdamW(model.parameters(), args.lr)
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    best_MAE, num_fail = 10e3, 1
    summaryWriter = SummaryWriter("./runs/MBSoc/")
    behav_adj_list = [model.graph_reconstruction(behav_index=b) for b in range(args.behav_num)]
    
    model_path = 'checkpoints/' + args.model_name + '/' + args.dataset + '_' + str(1-2*args.test_rate)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for ep in range(args.epoch):
        scheduler.step(epoch = ep)
        checkpoint_path = model_path + "/finetune_best_MBSRec_" + str(args.layers) + ".pt"
        output_information, total_loss = train_MBSoc(train_set, model, behav_adj_list, optimizer)
        logging.info(f'Epoch[{ep+1}/{args.epoch}] {output_information}')
        if ep % 5 == 0:
            MAE, RMSE = test_MBSoc(valid_set, model)
            if best_MAE > MAE:
                best_MAE = MAE
                torch.save(model, checkpoint_path)
                num_fail = 0
            elif num_fail < args.early_steps:
                num_fail += 1
            else:
                break
            summaryWriter.add_scalars(main_tag='Loss_Metrics', tag_scalar_dict={'Loss':total_loss,'MAE':MAE, 'RMSE': RMSE}, global_step=ep+1)
            logging.info("Epoch {} Validate: MAE: {:.4f}, RMSE: {:.4f}, best MAE: {:.4f}".format(ep+1, MAE, RMSE, best_MAE))
    summaryWriter.close()

    return best_MAE

def train_MBSoc(dataset, model, behav_adj_list, optimizer):
    model.train()
    total_loss = 0
    total_cl_loss = 0
    total_reg_loss = 0
    for _, batch in enumerate(next_batch_pairwise(dataset, args.batch)):
        user_idx, pos_idx, neg_idx, labels = batch
        user_idx = torch.tensor(user_idx, device=device)
        pos_idx = torch.tensor(pos_idx, device=device)
        neg_idx = torch.tensor(neg_idx, device=device)
        labels = torch.tensor(labels, dtype=torch.float32, device=device)
        main_loss, cl_loss, reg_loss = model.cal_loss([user_idx.long(), pos_idx.long()], labels, behav_adj_list)
        batch_loss = main_loss + args.cl_rate*cl_loss + reg_loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
        total_cl_loss += args.cl_rate*cl_loss.item()
        total_reg_loss += reg_loss.item()

    return f"loss:{total_loss:.3f} cl_loss:{total_cl_loss:.3f} reg_loss:{total_reg_loss:.3f} ", total_loss

def final_model_result(test_set, checkpoint_path):
    model = torch.load(checkpoint_path)
    MAE, RMSE = test_MBSoc(test_set, model)
    logging.info("Test: MAE: {:.4f}, RMSE: {:.4f}\n".format(MAE, RMSE))

if __name__ == '__main__':
    with open('dataset/' + args.dataset + '/MBSoc/all_data.pkl', 'rb') as f:
        (user_count, item_count, rating_count) = pickle.load(f)
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)
    
    fine_tune(user_count, item_count, train_set, valid_set)
    checkpoint_path_MAE = 'checkpoints/' + args.model_name + '/' + args.dataset + "_" + str(1-2*args.test_rate) + "/finetune_best_MBSRec_2.pt"
    final_model_result(test_set, checkpoint_path_MAE)

