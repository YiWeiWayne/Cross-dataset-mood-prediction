import numpy as np
from scipy.io import savemat

source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
save_path = '/mnt/data/Wayne'


Source_Train_X = np.load(save_path + '/' + source_dataset_name + '/Train_X.npy')
Target_Train_X = np.load(save_path + '/' + target_dataset_name + '/Train_X.npy')
savemat(save_path + '/' + source_dataset_name + '/Train_X.mat', {"Train_X": [Source_Train_X]})
savemat(save_path + '/' + target_dataset_name + '/Train_X.mat', {"Train_X": [Target_Train_X]})

