import torch
from torch import nn
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from scipy import signal
from numpy.fft import fft, fftfreq
import mne
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet

def read_csv(address, num):     # address代表读取地址，num代表当前的归类
    os.chdir(address)  # 设置工作目录
    file_chdir = os.getcwd()  # 获得工作目录
    for root, dirs, files in os.walk(file_chdir):  # os.walk会便利该目录下的所有文件
        for file in files:
            if os.path.splitext(file)[-1] == '.csv':  # 找csv文件
                filename_csv.append(file)  # 存储文件名
                file0 = (pd.read_csv(file))  # 打开当前文件
                data_lenth = len(file0)
                # print(data_lenth)
                raw = 0
                while raw < data_lenth:
                    if raw % 128 == 0:
                        seq = []
                        while raw < data_lenth:
                            tt = []
                            for i in choose_sensor:
                                tt.append(file0[sensor[i - 1]][raw])
                            seq.append(tt)
                            raw += 1
                            if raw < data_lenth and raw % data_len == 0:
                                seq = np.array(list(map(list, zip(*seq))))
                                # print('seq_shape: ' + str(seq.shape))
                                seq = dataPreprocess(seq)
                                kk = seq[:, ::second]
                                seq = kk
                                a = np.mean(seq)
                                D = np.var(seq)
                                seq = (seq - a) / np.sqrt(D)  # 中心化/标准化
                                # print('seq_shape: ' + str(seq.shape))
                                all_data.append(seq[None, :, :])
                                labels.append(num)
                                break
                    else:
                        raw += 1