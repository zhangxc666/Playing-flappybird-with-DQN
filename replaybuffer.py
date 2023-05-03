import pickle
import random
import numpy as np
import collections

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity, load=False, buffer_path="",save_directory=""):
        if not load:
            self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        else:
            try:
                self.buffer = pickle.load(open(buffer_path, 'rb'))
            except:
                print("缓冲区路径有误，请重新检查")
                exit(0)
        self.save_directory=save_directory

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def save(self,learning_rate,episodes,gamma): # 学习率，训练次数，折扣因子
        pickle.dump(self.buffer, open(f'{self.save_directory}/buffer_{episodes}_lr_{learning_rate}_gamma_{gamma}', 'wb'))
