import json
import numpy as np
import os


# melspec
with open("log_0_logs.json") as fb:
    data = json.load(fb)
index0 = 'train_loss_fake'
index1 = 'train_loss_dis'
index2 = 'val_R2_pearsonr'
index3 = 'val_pearsonr'
thr = 3
observe_range = 3
dis_thr = -5
fake_bias_min = 7
fake_bias_max = 8
dis_bias_min = 2
dis_bias_max = 3
observe_start = 0
observe_end = 2000
best_num = observe_end-observe_start
sort_index = np.argsort(-np.ones(len(data[index0][observe_start:observe_end]))*data[index0][observe_start:observe_end])

# print(np.asarray(data[index0])[425:435][:best_num])
# print(np.asarray(data[index1])[425:435][:best_num])
# print(np.asarray(data[index2])[425:435][:best_num])
# print(sort_index)

for ori_i in range(0, best_num):
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
    # if fake_p >= thr and data[index1][i] >= dis_thr and \
    #                         fake_bias_min <= np.abs(
    #                             data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
    #                         dis_bias_min <= np.abs(
    #                             data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
    #     print('1. Fake fool dis\n')
    #     print(str(ori_i) + '/' + str(i) + ': R2\n')
    #     print(data[index0][i - observe_range:i + 1])
    #     print(data[index1][i - observe_range:i + 1])
    #     print(data[index2][i - observe_range:i + 1])
    #     print(data[index3][i - observe_range:i + 1])
    # if fake_n >= thr and dis_p >= thr and data[index1][i] >= dis_thr and \
    #                         fake_bias_min <= np.abs(
    #                             data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
    #                         dis_bias_min <= np.abs(
    #                             data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
    #     print('2. dis good fake bad\n')
    #     print(str(ori_i) + '/' + str(i) + ': R2\n')
    #     print(data[index0][i - observe_range:i + 1])
    #     print(data[index1][i - observe_range:i + 1])
    #     print(data[index2][i - observe_range:i + 1])
    #     print(data[index3][i - observe_range:i + 1])
    # if fake_p >= thr and dis_p >= thr and data[index1][i] >= dis_thr and \
    #                         fake_bias_min <= np.abs(
    #                             data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
    #                         dis_bias_min <= np.abs(
    #                             data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
    #     print('3. dis and fake good\n')
    #     print(str(ori_i) + '/' + str(i) + ': R2\n')
    #     print(data[index0][i - observe_range:i + 1])
    #     print(data[index1][i - observe_range:i + 1])
    #     print(data[index2][i - observe_range:i + 1])
    #     print(data[index3][i - observe_range:i + 1])
    if fake_p >= thr and data[index1][i] >= dis_thr and \
                            fake_bias_min <= np.abs(
                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                            dis_bias_min <= np.abs(
                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
        print('4. Only fake good\n')
        print(str(ori_i) + '/' + str(i) + ': R2\n')
        print(data[index0][i - observe_range:i + 1])
        print(data[index1][i - observe_range:i + 1])
        print(data[index2][i - observe_range:i + 1])
        print(data[index3][i - observe_range:i + 1])
    if dis_p >= thr and data[index1][i] >= dis_thr and \
                            fake_bias_min <= np.abs(
                                data[index0][i] - data[index0][i - observe_range]) <= fake_bias_max and \
                            dis_bias_min <= np.abs(
                                data[index1][i] - data[index1][i - observe_range]) <= dis_bias_max:
        print('5. Only dis good\n')
        print(str(ori_i) + '/' + str(i) + ': R2\n')
        print(data[index0][i - observe_range:i + 1])
        print(data[index1][i - observe_range:i + 1])
        print(data[index2][i - observe_range:i + 1])
        print(data[index3][i - observe_range:i + 1])

