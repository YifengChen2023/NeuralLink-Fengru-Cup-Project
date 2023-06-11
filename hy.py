import socket
import time
import random
#IPV4,TCP协议
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#绑定ip和端口，bind接受的是一个元组
sock.bind(('127.0.0.1',8000))
#设置监听，其值阻塞队列长度，一共可以有5+1个客户端和服务器连接
sock.listen(5)

# 1.创建套接字
tcp_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# 2.准备连接服务器，建立连接
serve_ip = "127.0.0.1"
serve_port = 12220  #端口，比如8000
tcp_socket.connect((serve_ip,serve_port))  # 连接服务器，建立连接,参数是元组形式

a = [1, 2, 3, 4]
connection,address = sock.accept()
while True:
    # 将发送数据转化为String
    a = []
    for i in range(14):
        a.append(random.random())
    a.append(random.randint(1, 4))
    s=str(a)
    # 打印客户端地址
    print("client ip is:", address)
    # 接收数据,并存入buf
    buf = connection.recv(40960)
    print(buf.decode('utf-8'))

    # 发送数据
    connection.send(bytes(s, encoding="utf-8"))
    tcp_socket.send(str(a).encode('utf-8'))
    # 关闭连接
    # connection.close()
sock.close()