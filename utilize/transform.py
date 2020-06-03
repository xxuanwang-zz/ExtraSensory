import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class select_features_by_name(BaseEstimator, TransformerMixin):
    '''
    Select feature by given feature names. Compatible with sklearn transformer class.

    Keyword Arguments:
        feature_to_use: [list] -- feature names in list we want to select
        feature_names: [list] -- feature names of all the possible features 
    '''
    
    def __init__(self, feature_to_use, feature_names):

        self.feature_to_use = feature_to_use
        self.feature_names = feature_names
        
        fi = []
        for i, feature in enumerate(self.feature_names):
            if feature in self.feature_to_use:
                fi.append(i)
        self.fi = fi

    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[:, self.fi]

class select_features_by_sensors(BaseEstimator, TransformerMixin):

    '''
    Select feature by given sensor names. Compatible with sklearn transformer class.

    Keyword Arguments:
        sensors_to_use: [list] -- feature names in list we want to select
        feature_names: [list] -- feature names of all the possible features 
    '''

    def __init__(self, sensors_to_use, feature_names):
        self.sensors_to_use = sensors_to_use
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        fi = []
        for i, feature in enumerate(self.feature_names):
            if sensor_name_abbriviation[feature.split(':')[0]] in self.sensors_to_use:
                fi.append(i)
    
        return X[:, fi]
    
def select_target_labels(X, y, M, target_labels, label_names, drop_all_zero = True):
    '''
    Given target labels, labels matrix and all the possible label names, return label matrix with only target labels
    '''
    li = []
    for i, label_name in enumerate(label_names):
        if label_name in target_labels:
            li.append(i)

    sample_index = (np.sum(y[:, li], axis = 1) != 0)
    if drop_all_zero:
        return X[sample_index, :], y[sample_index, :][:, li], M[sample_index, :][:, li]
    else:
        return X[:,:], y[:, li], M[:, li]

def split_by_users(X, y, M, test_uuid, user_index):
    '''
    Split the dataset into training and test set given the test users' index in uuid_list
    '''
    X_train, y_train, M_train = [], [], []
    X_test, y_test, M_test = [], [], []

    for i in range(60):
        if i in test_uuid:
            X_test.append(X[user_index[i]:user_index[i+1], :])
            M_test.append(M[user_index[i]:user_index[i+1], :])
            y_test.append(y[user_index[i]:user_index[i+1]])
        else:
            X_train.append(X[user_index[i]:user_index[i+1], :])
            M_train.append(M[user_index[i]:user_index[i+1], :])
            y_train.append(y[user_index[i]:user_index[i+1]])

    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    M_train = np.concatenate(M_train)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)
    M_test = np.concatenate(M_test)
    
    return X_train, y_train, M_train, X_test, y_test, M_test


if __name__ == 'main': 

    None