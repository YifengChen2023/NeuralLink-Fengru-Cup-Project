from djitellopy import tello
import time

tello = tello.Tello() # 首先创建一个Tello的实例对象
tello.connect() # 连接到Tello无人机
# tello.takeoff() # Tello的起飞指令
while True:
    print(tello.get_battery())
    #tello.move()
    #tello.takeoff()  # Tello的起飞指令
    # tello.move_left(10) # Tello向左平移100厘米
    # tello.flip_left() # Tello在同一高度顺时针旋转90度
    # tello.flip_right()
    # tello.move_forward(10) # Tello向前飞行100厘米

    # tello.land() # Tello的降落指令

# 192.168.10.2