import os
import numpy as np


def load_data_to_numpy_array(path):
    # Input: Path to data folder
    
    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path,'test')

    train_files = get_file_names(test=False)
    test_files = get_file_names(test=True)
    
    Y_train = get_labels(os.path.join(train_path, 'y_train.txt'))
    Y_test = get_labels(os.path.join(test_path, 'y_test.txt'))

    X_train = get_array(train_path, train_files)
    X_test = get_array(test_path, test_files)
    return X_train, Y_train, X_test, Y_test


def get_file_names(test=False):
    extension = '.txt'
    names = ['body_acc_x_','body_acc_y_','body_acc_z_',
                'body_gyro_x_','body_gyro_y_','body_gyro_z_',
               'total_acc_x_','total_acc_y_','total_acc_z_']

    suffix = 'test' if test else 'train' 
    file_names = [x + suffix + extension for x in names]
    return file_names

    
def get_group(file):
    group = []
    with open(file) as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            group.append([float(i) for i in line.split(' ') if i != ''])
    return np.array(group)


def get_array(path, file_names):
    res = []
    for file_name in file_names:
        file = os.path.join(path, 'raw',file_name)
        res.append(get_group(file))

    res = np.array(res)
    return np.swapaxes(res,0,1)


def get_labels(file):
    with open(file) as f:
        lines = f.readlines()
        labels = [int(x[0].split('\n')[0])-1 for x in lines]
    return np.array(labels)


