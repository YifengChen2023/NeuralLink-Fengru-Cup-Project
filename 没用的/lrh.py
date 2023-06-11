import random
import pandas as pd
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
# from mne.viz import plot_filter, plot_ideal_filter
# from mne_connectivity import envelope_correlation
# from itertools import chain
# from Net import EEGNet

# get GPU or CPU device for training
device = "cuda" if torch.cuda.is_available() else "cpu"  # device is cuda
# device = "cpu"
save_dir = "E:/Pycharm2022/EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master/model/CNN_向左向右.pkl"  # 模型保存位置
filename_csv = []  # 文件名列表
second = 1
data_len = 128 * second  # 数据采样长度
all_data = []   # 训练数据
ans_data = []   # 对应训练数据的类别
labels = []
timeset = 'EEG.Counter'
'''设置which传感器数据是需要的'''
# choose_sensor = [1, 2, 3, 4, 7, 11, 12, 13, 14]
choose_sensor = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14] # 一共14个传感器
sensor = ['EEG.AF3', 'EEG.F7', 'EEG.F3', 'EEG.FC5', 'EEG.T7', 'EEG.P7', 'EEG.O1',
          'EEG.O2', 'EEG.P8', 'EEG.T8', 'EEG.FC6', 'EEG.F4', 'EEG.F8', 'EEG.AF4']     # 对应十四个传感器的label，方便读取数据


def dataPreprocess(raw_data):

    # print('raw_data.shape: ' + str(raw_data.shape))

    ch_names = ['AF3', 'F7', 'F3', 'FC5', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    # ch_names = ['AF3', 'F7', 'F3', 'FC5', 'O1', 'FC6', 'F4', 'F8', 'AF4']
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']
    # ch_types = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']

    sfreq = 128

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(raw_data, info)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)

    # 展示数据信息
    # print(raw)
    # print(raw.info)
    # raw.plot_sensors(ch_type='eeg', show_names=True)
    # raw.plot(duration=5, n_channels=len(raw.ch_names), scalings=dict(eeg=100e-6), clipping=None)
    # raw.plot_psd(average=True)
    # print('raw_data: ' + str(raw.get_data()))

    # FIR滤波
    # filter_params = mne.filter.create_filter(raw.get_data(), raw.info['sfreq'], l_freq=8, h_freq=30, fir_design='firwin')
    # mne.viz.plot_filter(filter_params, raw.info['sfreq'], flim=(0.1, 64))

    # raw_bandpass = raw.copy().filter(l_freq=8, h_freq=30, method='fir')
    # print('raw_bandpass_data: ' + str(raw_bandpass.get_data()))
    # print(raw_bandpass.get_data()[0])
    # with mne.viz.use_browser_backend('matplotlib'):
    #     fig = raw_bandpass.plot(duration=5, n_channels=len(raw.ch_names), scalings=dict(eeg=100e-6), clipping=None)
    #     fig.suptitle('Band-pass filtered at 8-30 Hz')
    # raw_bandpass.plot_psd(average=True)

    # 降采样
    raw_downsampled = raw.copy().resample(sfreq=50)
    # plt.show()

    return raw_downsampled.get_data()
    # ICA
    # ica = ICA(max_iter='auto')
    # ica.fit(raw_bandpass)
    # ica.plot_sources(raw_bandpass)
    # ica.plot_components()

    # 数据分段
    epochs = mne.make_fixed_length_epochs(raw_bandpass, duration=1, preload=False)
    # epochs.plot(scalings=dict(eeg=100e-6))
    # epochs.plot_psd(picks='eeg')

    # 数据叠加平均
    evoked = epochs.average()
    # 绘制逐导联的时序信号图
    # evoked.plot()
    # 绘制逐导联热力图
    # evoked.plot_image()
    # 绘制平均所有电极后的ERP
    # mne.viz.plot_compare_evokeds(evokeds=evoked, combine='mean')

    # 时频分析
    freqs = np.logspace(*np.log10([8, 30]), num=10)
    n_cycles = freqs / 2
    power = tfr_morlet(evoked, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False)
    # axe1 = plt.axes()
    # axe1.xaxis.set_visible(False)
    # axe1.yaxis.set_visible(False)

    # figs = power.plot(title='auto', colorbar=False, show=True)

    # figs[0].savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # figs[1].savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '1.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # fig.savefig('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', bbox_inches='tight', pad_inches=0, dpi=30)
    # img = cv2.imread('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', cv2.IMREAD_GRAYSCALE)
    # cv2.imwrite('waveletTransform/train/rh/rh' + str(fig_index)  + '.png', img)
    # plt.show()


# def one_hot(class_num, label):
#     ones = []
#     for i in range(class_num):
#         if i == label:
#             ones.append(1)
#         else:
#             ones.append(0)
#     return ones


'''开始读取数据'''
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

'''
#CNN
'''
def init_weight(module, non_linear):
    nn.init.kaiming_uniform_(module.weight, nonlinearity=non_linear)
    nn.init.zeros_(module.bias)

        # self.module = nn.ModuleDict({
        #     'conv1': nn.Conv2d(in_channels=14, out_channels=20, kernel_size=(1, 5)),
        #     'relu1': nn.ReLU(),
        #     'pool1': nn.MaxPool2d([1, 3]),
        #
        #     'conv2': nn.Conv2d(20, 40, kernel_size=(1, 5)),
        #     'tanh1': nn.Tanh(),
        #     'pool2': nn.MaxPool2d([1, 2]),
        #
        #     'conv3': nn.Conv2d(40, 60, kernel_size=(1, 3)),
        #     'relu2': nn.ReLU(),
        # })
        #
        # self.pool3 = nn.MaxPool2d([1, 2])
        # self.conv4 = nn.Conv2d(60, 80, (1, 3))
        # self.linear1 = nn.Linear(260, 2)


class CNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.ModuleDict({
            'conv1': nn.Conv2d(in_channels=13, out_channels=20, kernel_size=(1, 3)),
            'relu1': nn.ReLU(),
            'pool1': nn.MaxPool2d(kernel_size=(1, 2)),

            'conv2': nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1, 3)),
            'tanh1': nn.Tanh(),
            'pool2': nn.MaxPool2d(kernel_size=(1, 2)),

            'conv3': nn.Conv2d(in_channels=40, out_channels=60, kernel_size=(1, 3)),
            'relu2': nn.ReLU(),
        })
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 3))
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=(1, 3))
        self.linear1 = nn.Linear(260, 100)
        self.linear2 = nn.Linear(100, 2)
        self.softmax = nn.Softmax()

        init_weight(self.module['conv1'], 'relu')
        init_weight(self.module['conv2'], 'tanh')
        init_weight(self.module['conv3'], 'relu')
        init_weight(self.conv4, 'tanh')
        init_weight(self.linear1, 'tanh')
        init_weight(self.linear2, 'relu')

    def forward(self, x):
        for key in self.module:
            x = self.module[key](x)
        X_pool = self.pool3(x)
        X_conv = torch.tanh(self.conv4(X_pool))
        # DeepMF
        x = torch.cat((X_pool.flatten(start_dim=1), X_conv.flatten(start_dim=1)), dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    # read_csv("E:\Pycharm2022\EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master\lrh\向左想象", 0)
    # read_csv('E:\Pycharm2022\EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master\lrh\向右想象', 1)
    # read_csv('E:\Pycharm2022\EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master\lrh\起飞想象', 0)
    # read_csv("E:\Pycharm2022\EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master\lrh\降落想象", 1)
    # read_csv("E:\Pycharm2022\EEG-Motor-Imagery-Classification-CNNs-TensorFlow-master\lrh\静息睁眼", 2)

    # print(all_data)

    size = (np.array(all_data)).shape
    print('all_data_shape: ' + str(size))
    all_data = np.array(all_data)
    lenth = len(all_data)
    print('共有数据集{}条'.format(lenth))


    seed = random.randint(1, 100)
    print('随机种子为 {: d}'.format(seed))
    x_data = all_data
    y_label = labels
    np.random.seed(seed)
    np.random.shuffle(x_data)
    np.random.seed(seed)
    np.random.shuffle(y_label)
    x_train = torch.tensor(x_data[:-int(0.3*lenth)], dtype=torch.float)
    y_train = torch.tensor(y_label[:-int(0.3*lenth)], dtype=torch.int64)
    x_val = torch.tensor(x_data[-int(0.3*lenth):-int(0.2*lenth)], dtype=torch.float)
    y_val = torch.tensor(y_label[-int(0.3*lenth):-int(0.2*lenth)], dtype=torch.int64)
    x_test = torch.tensor(x_data[-int(0.3*lenth):], dtype=torch.float)
    y_test = torch.tensor(y_label[-int(0.3*lenth):], dtype=torch.int64)
    x_train = x_train.permute(0, 2, 1, 3)
    x_val = x_val.permute(0, 2, 1, 3)
    x_test = x_test.permute(0, 2, 1, 3)
    print('x_train.shape: ' + str(x_train.shape))
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    #创建对象
    model = CNNetwork().to(device)
    # model = EEGNet().to(device)

    #损失函数和ADAM
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    #初始化列表储存准确度和损失
    accuracy = []
    loss_re = []
    acc_max = 0     # 用来储存训练中的验证集准确率的max
    loss_min = 10
    EPOCHS = 1000
    tttt = 0
    for epoch in range(EPOCHS):
        model.train()
        pre_train = model(x_train)
        loss = loss_fn(pre_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            correct = 0
            pre_val = model(x_val)
            y = torch.argmax(pre_val, dim=1)
            # pp = torch.max(pre_val, dim=1)
            # qq = torch.min(pre_val, dim=1)
            ttt = 0
            for i in range(len(y_val)):
                ttt += 1
                if y_val[i] == y[i]:
                    correct += 1
            # acc = correct/len(y_val)*100
            if ttt == 0:
                acc = 0
            else:
                acc = correct / ttt * 100
            print('num = {:d}, loss = {: .6f}, val_acc = {: .3f}'.format(ttt, float(loss), acc))
            loss_re.append(float(loss))
            accuracy.append(acc)
            if epoch > EPOCHS/2 and (tttt < ttt or (tttt == ttt and acc_max < acc) or (tttt == ttt and acc_max == acc and loss_min > loss)):
                file = open(save_dir, 'wb')  # 保存网络
                pickle.dump(model, file)
                file.close()
                tttt = ttt
                acc_max = acc
                loss_min = float(loss)
    '''
    #读取验证集效果最佳的模型
    '''
    # file = open(save_dir, 'rb')
    # model = pickle.load(file)
    # file.close()
    print('选取训练过程中验证集效果最佳的参数\n此时tttt = {:d}, loss={:.6f}\t验证集acc={:.3f}'.format(tttt, loss_min, acc_max))
    print('****start testing****')
    # model.eval()
    correct = 0
    pre_test = model(x_test)
    pp = torch.max(pre_test, dim=1)
    qq = torch.min(pre_test, dim=1)
    y = torch.argmax(pre_test, dim=1)
    ttt = 0
    for i in range(len(y_test)):
        ttt += 1
        if y_test[i] == y[i]:
            correct += 1
    print('最终num = {:d}, test_acc = {: .3f}'.format(ttt, correct/ttt*100))

    '''
    #可视化
    '''
    plt.figure(1)
    plt.title('loss')
    line0, = plt.plot(loss_re)
    plt.legend(handles=[line0], labels=["train-loss"], loc='best')
    plt.figure(2)
    plt.title('acc')
    line1, = plt.plot(accuracy)
    line2 = plt.hlines(correct/ttt*100, 0, EPOCHS // 5, color="red")
    plt.legend(handles=[line1, line2], labels=["val_acc", "test_acc"], loc='best')
    plt.show()
