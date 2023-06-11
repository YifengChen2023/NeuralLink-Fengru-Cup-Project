import socket
import time
import random
#IPV4,TCP协议
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#绑定ip和端口，bind接受的是一个元组
sock.bind(('127.0.0.1',12220))
#设置监听，其值阻塞队列长度，一共可以有5+1个客户端和服务器连接
sock.listen(5)

connection,address = sock.accept()

while 1:
    buf = connection.recv(40960)
    print(buf.decode('utf-8'))
sock.close()