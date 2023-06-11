import socket
import threading

# 设置参数
ip = ''  # 监听所有网络接口
port = 12345

# 创建socket连接
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((ip, port))
s.listen()

# 定义存储数据的列表和锁
data_list = []
lock = threading.Lock()

# 数据接收和存储线程
def recv_thread():
    while True:
        conn, addr = s.accept()
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                # 存储接收到的数据
                with lock:
                    data_list.append(data.decode())

# 数据处理和执行线程
def process_thread():
    executed_set = set()  # 存储已执行过的数据的集合
    while True:
        # 获取最新的未执行过的数据
        with lock:
            data = None
            for i in range(len(data_list)-1, -1, -1):
                if data_list[i] not in executed_set:
                    data = data_list.pop(i)
                    break
        if data is not None:
            # 执行数据
            print(data)  # 替换为实际的数据处理和执行代码
            # 标记已执行
            executed_set.add(data)

# 启动线程
recv_t = threading.Thread(target=recv_thread)
recv_t.start()

process_t = threading.Thread(target=process_thread)
process_t.start()

# 等待线程结束
recv_t.join()
process_t.join()
