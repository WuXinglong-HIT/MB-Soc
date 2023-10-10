import os
import pickle
import pandas as pd
import numpy as np
from Params import args
import scipy.sparse as sp

# Load data
path = 'dataset/' + args.dataset
rating_path = path + '/rating_data.txt'
trust_path = path + '/trust_data.txt'
save_path = path + '/' + args.model_name
if not os.path.exists(save_path):
    os.makedirs(save_path)

if args.dataset == 'Epinions':
    rating_df = pd.read_csv(rating_path, sep=' ', header=None, error_bad_lines=False )
    rating_df = rating_df.astype(int)
    list=[1, 2, 3, 4, 5]
    rating_df = rating_df[rating_df[2].isin(list)]        # make sure the rating is in [1,2,3,4,5]
    rating_df.columns = ['uid', 'iid', 'rating']

    trust_df = pd.read_csv(trust_path, sep=' ', header=None, error_bad_lines=False )
    trust_df = trust_df[trust_df[3] == 1]
    trust_df = trust_df.iloc[:, [1,2]]
    trust_df = trust_df.astype(int)
    trust_df.columns = ['source_uid', 'target_uid']
    print("Epinions: Successfully load the rating data and trust data......")

elif args.dataset == 'Ciao':
    rating_df = pd.read_csv(rating_path, sep='  ', header=None, error_bad_lines=False )
    rating_df = rating_df.iloc[:, [0,1,3]]
    rating_df = rating_df.astype(int)
    list=[1, 2, 3, 4, 5]
    rating_df = rating_df[rating_df[3].isin(list)]        # make sure the rating is in [1,2,3,4,5]
    rating_df.columns = ['uid', 'iid', 'rating']

    trust_df = pd.read_csv(trust_path, sep='  ', header=None, error_bad_lines=False )
    trust_df = trust_df.iloc[:, [0,1]]
    trust_df = trust_df.astype(int)
    trust_df.columns = ['source_uid', 'target_uid']
    print("Ciao: Successfully load the rating data and trust data......")

else:
    print("Dataset {} is not supported!".format(args.dataset))
    

def loadDataSet():
    global rating_df, trust_df
    # Collect data characteristics
    rating_uid_list = np.sum(np.array(rating_df.iloc[:, [0]]).tolist(), axis=1)
    rating_iid_list = np.sum(np.array(rating_df.iloc[:, [1]]).tolist(), axis=1)
    rating_list = np.sum(np.array(rating_df.iloc[:, [2]]).tolist(), axis=1)
    trust_list_1 = max(np.sum(np.array(trust_df.iloc[:, [0]]).tolist(), axis=1)) + 1
    trust_list_2 = max(np.sum(np.array(trust_df.iloc[:, [1]]).tolist(), axis=1)) + 1
    rating_uid_0 = max(rating_uid_list) + 1
    user_count = int(max(trust_list_1, trust_list_2, rating_uid_0))   # the number of users
    item_count = int(max(rating_iid_list) + 1)                        # the number of items
    rating_count = int(max(rating_list) + 1)                          # the number of ratings

    # Generate negative item for each user
    rating_matrix = sp.coo_matrix((rating_list, (rating_uid_list, rating_iid_list)), shape=(user_count, item_count))
    rating_matrix = rating_matrix.toarray()
    pos_iids = []
    for i in range(user_count):
        pos_iids.append(np.where(rating_matrix[i]!=0)[0])
    print("Successfully generate positive items for each user.")
    
    with open(save_path+ '/pos_items.pkl', 'wb') as f:
        pickle.dump(pos_iids, f, pickle.HIGHEST_PROTOCOL)

    # Generate user-user trust graph
    source_list = np.sum(np.array(trust_df.iloc[:, [0]]).tolist(), axis=1)
    target_list = np.sum(np.array(trust_df.iloc[:, [1]]).tolist(), axis=1)
    assert len(source_list) == len(target_list)
    with open(save_path + '/social_relation.pkl', 'wb') as f:
        pickle.dump(source_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(target_list, f, pickle.HIGHEST_PROTOCOL)

    # Generate the train, valid, test set
    rating_df = rating_df.sample(frac=1).reset_index(drop=True)      # disorder the dataset
    test_count = int(len(rating_list) * float(args.test_rate))
    convert_dict = {'uid': int, 'iid': int, 'rating': float}
    rating_df = rating_df.astype(convert_dict)
    test_df = rating_df[0:test_count]
    valid_df = rating_df[test_count:2*test_count]
    train_df = rating_df[2*test_count:]
    print("The size of train, valid, test set is ", len(train_df), ", ", len(valid_df), ", ", len(test_df))

    # Generate single-behavior data
    train_set_b0, train_set_b1, train_set_b2 = [], [], []
    train_set = np.array(train_df).tolist()
    valid_set = np.array(valid_df).tolist()
    test_set = np.array(test_df).tolist()
    for train_data in train_set:
        if train_data[2]>0 and train_data[2]<3:
            train_set_b0.append(train_data)
        if train_data[2]==3:
            train_set_b1.append(train_data)
        if train_data[2]>3 and train_data[2]<=5:
            train_set_b2.append(train_data)
    print("Successfully single-behavior interaction data.")
    with open(save_path + '/behavior_train_set.pkl', 'wb') as f:
        pickle.dump(train_set_b0, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_set_b1, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_set_b2, f, pickle.HIGHEST_PROTOCOL)
        
    return user_count, item_count, rating_count, train_set, valid_set, test_set


if __name__ == '__main__':
    user_count, item_count, rating_count, train_set, valid_set, test_set = loadDataSet()    
    with open(save_path + '/all_data.pkl', 'wb') as f:
        pickle.dump((user_count, item_count, rating_count), f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    print("Load Over!!")

