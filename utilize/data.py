import gzip
import io as StringIO
import os 

import numpy as np
import pandas as pd

def parse_header_of_csv(csv_str):
    # Isolate the headline columns:
    headline = csv_str[:csv_str.index('\n')]
    columns = headline.split(',')

    # The first column should be timestamp:
    assert columns[0] == 'timestamp'
    # The last column should be label_source:
    assert columns[-1] == 'label_source'
    
    # Search for the column of the first label:
    for (ci,col) in enumerate(columns):
        if col.startswith('label:'):
            first_label_ind = ci
            break
        pass

    # Feature columns come after timestamp and before the labels:
    feature_names = columns[1:first_label_ind]
    # Then come the labels, till the one-before-last column:
    label_names = columns[first_label_ind:-1]
    for (li,label) in enumerate(label_names):
        # In the CSV the label names appear with prefix 'label:', but we don't need it after reading the data:
        assert label.startswith('label:')
        label_names[li] = label.replace('label:','')
        pass
    
    return (feature_names,label_names)

def parse_body_of_csv(csv_str,n_features):
    # Read the entire CSV body into a single numeric matrix:
    full_table = np.loadtxt(StringIO.StringIO(csv_str),delimiter=',',skiprows=1)
    
    # Timestamp is the primary key for the records (examples):
    timestamps = full_table[:,0].astype(int)
    
    # Read the sensor features:
    X = full_table[:,1:(n_features+1)]
    
    # Read the binary label values, and the 'missing label' indicators:
    trinary_labels_mat = full_table[:,(n_features+1):-1] # This should have values of either 0., 1. or NaN
    M = np.isnan(trinary_labels_mat) # M is the missing label matrix
    Y = np.where(M,0,trinary_labels_mat) > 0. # Y is the label matrix
    
    return (X,Y,M,timestamps)

'''
Read the data (precomputed sensor-features and labels) for a user.
This function assumes the user's data file is present.
'''
def read_user_data(uuid):
    user_data_file = 'user_data/%s.features_labels.csv.gz' % uuid

    # Read the entire csv file of the user:
    with gzip.open(user_data_file,'rb') as fid:
        csv_str = fid.read().decode()
        pass

    (feature_names,label_names) = parse_header_of_csv(csv_str)
    n_features = len(feature_names)
    (X,Y,M,timestamps) = parse_body_of_csv(csv_str,n_features)

    return (X,Y,M,timestamps,feature_names,label_names)


'''
Load the data of all the users 
'''

UUID_LIST = []

for file in os.listdir('user_data'):
    if file.split('.')[-1] == 'gz':
        UUID_LIST.append(file.split('.')[0])
print('Found %d users data.' %(len(UUID_LIST)))

def load_all_data(uuid_list = UUID_LIST):
    '''
    Load data from all the sixty users

    Returns:
        X: [narray] -- feature matrix in shape of [n_smaples, n_features]
        y: [narray] -- label matrix in shape of [n_smaples, n_labels]
        M: [narray] -- missing label matrix, element is false means that the label is missing
        user_index: [list] -- indicate the starting index of each user data
        feature_names: [list] -- feature names of all the possible features
        label_names: [list] -- label names of all the possible labels
    '''

    X, y, M = [], [], []
    user_index = [0]

    for i, uuid in enumerate(uuid_list):

        X_i,y_i,M_i,timestamps,feature_names,label_names = read_user_data(uuid)
        user_index.append(user_index[i]+X_i.shape[0])

        X.append(X_i)
        y.append(y_i)
        M.append(M_i)
        
    X = np.concatenate(X)
    y = np.concatenate(y)
    M = np.concatenate(M)

    return X, y, M, user_index, feature_names, label_names

if __name__ == 'main': 

    # load the data of the first user in the list
    X,Y,M,timestamps,feature_names,label_names = read_user_data(UUID_LIST[0])

    # Load the data of all the users 
    X, y, M, user_index, feature_names, label_names = load_all_data()
