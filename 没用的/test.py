import time
from socket import *
'''from pycrazyswarm import Crazyswarm
import matplotlib.pyplot as plt'''

TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0

from math import sqrt

import numpy as np
# from pycrazyswarm import *

Z = 1.0
sleepRate = 30


def goCircle(timeHelper, cf, totalTime, radius, kPosition):
    startTime = timeHelper.time()
    pos = cf.position()
    startPos = cf.initialPosition + np.array([0, 0, Z])
    center_circle = startPos - np.array([radius, 0, 0])
    while True:
        time = timeHelper.time() - startTime
        if time > totalTime:
            break
        omega = 2 * np.pi / totalTime
        vx = -radius * omega * np.sin(omega * time)
        vy = radius * omega * np.cos(omega * time)
        desiredPos = center_circle + radius * np.array(
            [np.cos(omega * time), np.sin(omega * time), 0])
        errorX = desiredPos - cf.position()
        cf.cmdVelocityWorld(np.array([vx, vy, 0] + kPosition * errorX), yawRate=0)
        timeHelper.sleepForRate(sleepRate)


def goTo_nonRelative(allcfs, timeHelper, relative_pos=None, target_time=2):
    # allcfs, timeHelper = setUp()
    if relative_pos is None:
        relative_pos = [1, 1, 1]

    for cf in allcfs.crazyflies:
        pos = np.array(cf.initialPosition) + np.array(relative_pos)
        cf.goTo(pos, 0, target_time)
    timeHelper.sleep(2.0)

    for cf in allcfs.crazyflies:
        pos = cf.initialPosition + np.array(relative_pos)
        assert np.all(np.isclose(cf.position(), pos))


def goTriangle(timeHelper, allcfs, height, totalTime, length):
    startTime = timeHelper.time()
    pos1 = np.array([0, 0, height])
    pos2 = np.array([length, 0, height])
    pos3 = np.array([1 / 2 * length, sqrt(3) / 2 * length, height])

    pre_time = totalTime / 3
    goTo_nonRelative(allcfs=allcfs, timeHelper=timeHelper, relative_pos=pos2, target_time=pre_time)
    goTo_nonRelative(allcfs=allcfs, timeHelper=timeHelper, relative_pos=pos3, target_time=pre_time)
    goTo_nonRelative(allcfs=allcfs, timeHelper=timeHelper, relative_pos=pos1, target_time=pre_time)


def go_and_back(timeHelper, allcfs, height, totalTime, length):
    startTime = timeHelper.time()
    pos1 = np.array([0, 0, height])
    pos2 = np.array([length, 0, height])
    # pos3 = np.array([1 / 2 * length, sqrt(3) / 2 * length, height])

    pre_time = totalTime / 2
    goTo_nonRelative(allcfs=allcfs, timeHelper=timeHelper, relative_pos=pos2, target_time=pre_time)
    goTo_nonRelative(allcfs=allcfs, timeHelper=timeHelper, relative_pos=pos1, target_time=pre_time)


# 创建套接字
tcp_server = socket(AF_INET, SOCK_DGRAM)
# 绑定ip，port
# 这里ip默认本机
address = ('', 5353)
tcp_server.bind(address)
# 启动被动连接
# 多少个客户端可以连接
tcp_server.listen(128)
# 使用socket创建的套接字默认的属性是主动的
# 使用listen将其变为被动的，这样就可以接收别人的链接了
# 创建接收
# 如果有新的客户端来链接服务器，那么就产生一个新的套接字专门为这个客户端服务
client_socket, clientAddr = tcp_server.accept()
'''client_socket用来为这个客户端服务，相当于的tcp_server套接字的代理
tcp_server_socket就可以省下来专门等待其他新客户端的链接
这里clientAddr存放的就是连接服务器的客户端地址'''
'''swarm = Crazyswarm()
timeHelper = swarm.timeHelper
allcfs = swarm.allcfs
allcfs.takeoff(targetHeight=1.0, duration=TAKEOFF_DURATION)'''

while (1):
    # 接收对方发送过来的数据
    from_client_msg = int(client_socket.recv(1))
    if from_client_msg == None:
        break
    if from_client_msg == 1:
        print('1')
        # go_and_back(timeHelper, allcfs, totalTime=3, length=2, height=1.0)
        # goCircle(timeHelper, cf, totalTime=6, radius=1, kPosition=1)
        # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    elif from_client_msg == 0:
        print('0')
        # goTriangle(timeHelper, allcfs, totalTime=3, length=3, height=1.0)
        # cf.cmdVel(roll = '1.3')
        # timeHelper.sleep(TAKEOFF_DURATION + HOVER_DURATION)
    # plt.close('all') #关闭所有 figure windows
    # timeHelper.sleep(TAKEOFF_DURATION)

    now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    # 发送数据给客户端
    # send_data = client_socket.send((str(now_time)+" 收到动作，动作" + str(from_client_msg).decode("gbk") + "开始执行").encode("gbk"))
    # 关闭套接字
    # 关闭为这个客户端服务的套接字，就意味着为不能再为这个客户端服务了
    # 如果还需要服务，只能再次重新连
client_socket.close()
# allcfs.land(targetHeight=0.04, duration=2.5)