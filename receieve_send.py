"""Example program to show how to read a multi-channel time series from LSL."""
import time
from torch import nn
import torch
import numpy as np
from pylsl import StreamInlet, resolve_stream
from socket import *
import time
import joblib
from data_preprocess import dataPreprocess


def init_weight(module, non_linear):
    nn.init.kaiming_uniform_(module.weight, nonlinearity=non_linear)
    nn.init.zeros_(module.bias)


class ConvNets_FFT(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.module = nn.ModuleDict({
            'conv1': nn.Conv2d(8, 20, [1, 5]),
            'relu1': nn.ReLU(),
            'bn1': nn.BatchNorm2d(20),
            'pool1': nn.MaxPool2d([1, 2]),

            'conv2': nn.Conv2d(20, 40, [1, 3]),
            'bn2': nn.BatchNorm2d(40),
            'tanh1': nn.Tanh(),
            'pool2': nn.MaxPool2d([1, 2]),

            'conv3': nn.Conv2d(40, 60, [1, 3]),
            'relu2': nn.ReLU(),
            'bn3': nn.BatchNorm2d(60),
            'pool3': nn.MaxPool2d([1, 3]),

            'conv4': nn.Conv2d(60, 80, [1, 3], stride=[1, 3]),
            'bn4': nn.BatchNorm2d(80),
            'tanh2': nn.Tanh(),
        })
        self.fft_module = nn.ModuleDict({
            'conv1': nn.Conv2d(8, 16, [1, 5], stride=[1, 2]),
            'relu1': nn.ReLU(),
            'bn1': nn.BatchNorm2d(16),

            'conv2': nn.Conv2d(16, 16, [1, 3], stride=[1, 2]),
            'bn2': nn.BatchNorm2d(16),
            'tanh1': nn.Tanh(),
            'pool1': nn.MaxPool2d([1, 2])
        })
        self.dropout = nn.Dropout(p=0.4)
        self.linear1 = nn.Linear(240, num_class)

        init_weight(self.module['conv1'], 'relu')
        init_weight(self.module['conv2'], 'tanh')
        init_weight(self.module['conv3'], 'relu')
        init_weight(self.module['conv4'], 'tanh')
        init_weight(self.fft_module['conv1'], 'relu')
        init_weight(self.fft_module['conv2'], 'tanh')
        init_weight(self.linear1, 'relu')

    def forward(self, x):
        fft = torch.fft.fft(x, 88).real[:, :, :, 0:45]
        for key in self.module:
            x = self.module[key](x)
        for key in self.fft_module:
            fft = self.fft_module[key](fft)

        x = torch.cat((x.flatten(start_dim=1), fft.flatten(start_dim=1)), dim=1)
        x = self.dropout(x)
        x = self.linear1(x)
        x = torch.nn.functional.softmax(x, dim=1)
        return x


if __name__ == '__main__':
    # print("looking for a stream...")
    # first resolve a Motion stream on the lab network
    streams = resolve_stream('type',
                             'EEG')  # You can try other stream types such as: EEG, EEG-Quality, Contact-Quality, Performance-Metrics, Band-Power
    # print(streams)

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])
    ist1, ist2 = 0, 0
    percent_data = []
    final_data = []
    action2id = {'空': 0, '起飞': 1, '向左': 2, '向右': 3, '降落': 4, '舌动': 5, '眼动': 6}
    id2action = {0: '空', 1: '起飞', 2: '向左', 3: '向右', 4: '降落', 5: '舌动', 6: '眼动'}

    model = joblib.load(filename='E:\junior_1\冯如杯\\april-25眼动静息\静息眼动0425.pkl')

    device = "cuda" if torch.cuda.is_available() else "cpu"  # device is cuda

    # 1.创建套接字
    # tcp_socket = socket(AF_INET,SOCK_STREAM)
    # # 2.准备连接服务器，建立连接
    # serve_ip = "192.168.31.211"
    # serve_port = 12220  #端口，比如8000
    # tcp_socket.connect((serve_ip,serve_port))  # 连接服务器，建立连接,参数是元组形式
    test_data = []
    start_time = time.time()
    while True:
        # Returns a tuple (sample,timestamp) where sample is a list of channel values and timestamp is the capture time of the sample on the remote machine,
        # or (None,None) if no new sample was available
        sample, timestamp = inlet.pull_sample()
        if timestamp is not None:
            # print(timestamp, sample)
            data = sample[3:7] + sample[13:17]
            # print('data: ' + str(data))
            percent_data.append(data)
            ist1 += 1
            if ist1 == 128:

                ist1 = 0
                percent_data = np.array(percent_data).T
                # print(percent_data.shape)
                seq = dataPreprocess(percent_data)
                # print(percent_data2.shape)
                a = np.mean(seq)
                D = np.var(seq)
                seq = (seq - a) / np.sqrt(D)
                test_data.append(seq[None, :, :])
                percent_data = []

                ist2 += 1

                if ist2 == 5:
                    ist2 = 0
                    # print(test_data)
                    test_data = np.array(test_data)
                    test_data = torch.tensor(test_data, dtype=torch.float)
                    test_data = test_data.permute(0, 2, 1, 3)
                    test_data = test_data.to(device)
                    pre = model(test_data)
                    # print(pre)
                    label = torch.argmax(pre, dim=1)

                    order = []
                    length = len(label)
                    for i in range(length):
                        if label[i] == 0:
                            order.append(action2id['空'])
                        elif label[i] == 1:
                            order.append(action2id['眼动'])
                    test_data = []
                    print('order: ' + str(order))
                    # 输出最终的指令id
                    instructId = max(order, key=order.count)
                    print('ins: ' + str(id2action[instructId]))
                    # tcp_socket.send(str(instructId).encode('utf-8'))
                    # print(f'cost:{time.time()-start_time:.4f}s')
                    # print('send')
                    # from_server_msg = tcp_socket.recv(200)
                    # 加上.decode("gbk")可以解决乱码
                    # print(from_server_msg.decode("gbk"))
        else:
            # tcp_socket.close()
            break
