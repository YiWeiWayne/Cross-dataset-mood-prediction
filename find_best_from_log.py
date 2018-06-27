import json
import numpy as np
import os

source_dataset_name = 'AMG_1608'
target_dataset_name = 'CH_818'
source_data_num = 1608
target_data_num = 818
algorithm = ['WADDA']
save_path = '/mnt/data/Wayne'
emotions = ['valence', 'arousal']
features = ['melSpec_lw', 'pitch+lw', 'rCTA']
actions = ['melSpec_lw', 'pitch+lw', 'rCTA']
pretrain_path = dict(zip(algorithm, [['', '', '']]))
pretrain_path[algorithm[0]] = [
        save_path + '/(' + actions[0] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1045.07',
        save_path + '/(' + actions[1] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1150.00',
        save_path + '/(' + actions[2] + ')' + algorithm[0] + '_S_' + source_dataset_name + '_T_'
        + target_dataset_name + '_20180623.1153.14']
epoch_s = [['', '', ''], ['', '', '']]
epoch_e = [['', '', ''], ['', '', '']]
th = [['', '', ''], ['', '', '']]
dis_th = [['', '', ''], ['', '', '']]
f_bias_min = [['', '', ''], ['', '', '']]
f_bias_max = [['', '', ''], ['', '', '']]
d_bias_max = [['', '', ''], ['', '', '']]
d_bias_min = [['', '', ''], ['', '', '']]
R2 = [['', '', ''], ['', '', '']]
p = [['', '', ''], ['', '', '']]
observer_target = [['', '', ''], ['', '', '']]
epoch_s = [[0, 0, 0], [0, 0, 0]]
epoch_e = [[2000, 2000, 500], [2000, 2000, 500]]
th = [[3, 3, 2], [4, 3, 3]]
dis_th = [[-5, -2, -5], [0, -5, -5]]
f_bias_min = [[3, 5, 9], [10, 7, 4]]
f_bias_max = [[4, 6, 10], [15, 8, 5]]
d_bias_min = [[0, 0, 1], [0, 2, 0]]
d_bias_max = [[5, 1, 2], [5, 3, 5]]
observe_target = [['dis', 'fake', 'fake'], ['dis', 'dis', 'dis']]
index0 = 'train_loss_fake'
index1 = 'train_loss_dis'
index2 = 'val_R2_pearsonr'
index3 = 'val_pearsonr'
for e in range(0, len(emotions)):
    for f in range(0, len(features)):
        if os.path.exists(pretrain_path[algorithm[0]][f] + '/' + emotions[e] + '/log_0_logs.json'):
            with open(pretrain_path[algorithm[0]][f] + '/' + emotions[e] + '/log_0_logs.json', "r") as fb:
                data = json.load(fb)
            thr = th[e][f]
            dis_thr = dis_th[e][f]
            observe_range = th[e][f]
            fake_bias_min = f_bias_min[e][f]
            fake_bias_max = f_bias_max[e][f]
            dis_bias_min = d_bias_min[e][f]
            dis_bias_max = d_bias_max[e][f]
            observe_start = epoch_s[e][f]
            observe_end = epoch_e[e][f]
            best_num = epoch_e[e][f]-epoch_s[e][f]
            sort_index = np.argsort(
                -np.ones(len(data[index0][observe_start:observe_end])) * data[index0][observe_start:observe_end])
            skip_key = False
            ori_i = -1
            while (not skip_key) and (ori_i != best_num-1):
                ori_i = ori_i + 1
                # print(ori_i)
                i = observe_start + sort_index[ori_i]
                fake_p = 0
                dis_p = 0
                fake_n = 0
                dis_n = 0
                for j in range(i - observe_range, i):
                    if data[index0][j + 1] > data[index0][j]:
                        fake_p = fake_p + 1
                    if data[index1][j + 1] < data[index1][j]:
                        dis_n = dis_n + 1
                    if data[index0][j + 1] < data[index0][j]:
                        fake_n = fake_n + 1
                    if data[index1][j + 1] > data[index1][j]:
                        dis_p = dis_p + 1
                if observe_target[e][f] == 'dis':
                    if dis_p >= thr and data[index1][i] >= dis_thr and \
                                            fake_bias_min <= np.abs(
                                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                                            dis_bias_min <= np.abs(
                                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
                        skip_key = True
                        R2[e][f] = data[index2][i]
                        p[e][f] = data[index3][i]
                        print(emotions[e] + '/' + features[f] + '/' + str(i) + '/' + str(R2[e][f]) + '/' + str(p[e][f]))
                elif observe_target[e][f] == 'fake':
                    if fake_p >= thr and data[index1][i] >= dis_thr and \
                                            fake_bias_min <= np.abs(
                                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                                            dis_bias_min <= np.abs(
                                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
                        skip_key = True
                        R2[e][f] = data[index2][i]
                        p[e][f] = data[index3][i]
                        print(emotions[e] + '/' + features[f] + '/' + str(i) + '/' + str(R2[e][f]) + '/' + str(p[e][f]))
            else:
                print('OK/' + str(ori_i))