import socket
import os
import random
import pandas as pd
import numpy as np

# data = pd.read_csv(r'C:\Users\Lenovo User\Desktop\冯如杯-脑机交互\方法汇总\静息\train0405_rest_EPOCX_171453_2023.04.05T09.27.55+08.00.csv')
# data = np.array(data)
# data0 = data[:, 4:18]
# print(data0.shape)
# data0 = data0.tolist()
# print(type(data0))


#IPV4,TCP协议
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#绑定ip和端口，bind接受的是一个元组
sock.bind(('127.0.0.1',8000))
#设置监听，其值阻塞队列长度，一共可以有5+1个客户端和服务器连接
sock.listen(5)

# # 1.创建套接字
# tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# # 2.准备连接服务器，建立连接
# serve_ip = "127.0.0.1"
# serve_port = 12220  #端口，比如8000
# tcp_socket.connect((serve_ip,serve_port))  # 连接服务器，建立连接,参数是元组形式

a = [1, 2, 3, 4]
t = 0
struc = random.randint(1, 4)
connection, address = sock.accept()
struc = 0
count = 0
while True:
    # 将发送数据转化为String
    a = []
    b = []
    c = []
    M = [4309.7886574891, 4298.7697202038335, 4330.342540854722, 4326.737623612304, 4290.042517634397, 4290.283844391647,
         4294.674507826712, 4305.180294637339, 4282.67872789123, 4271.779960641943, 4486.425850332993, 4421.121165935583,
         4384.4559363871585, 4445.615973379307]
    for i in range(14):
        a.append(random.uniform(-50, 50) + M[i])
        b.append(random.uniform(-50, 50) + M[i])

    # a = data0[count]
    # b = data0[count + 1]
    # count += 2

    t = t + 1
    if t > 128:
        t = 0
    a.append(struc)
    b.append(struc)
    b.append(t * 10)
    c = a+b
    s = str(c)
    print(c)
    # 打印客户端地址
    # print("client ip is:", address)
    # 接收数据,并存入buf
    buf = connection.recv(40960)
    info = buf.decode('utf-8')
    if 'switch' in info:
        print('switch=', info[7])
    elif 'mode' in info:
        print('mode=', info[5])
    # 发送数据
    connection.send(bytes(s, encoding="utf-8"))
    # tcp_socket.send(str(a).encode('utf-8'))
    # 关闭连接
    # connection.close()
sock.close()