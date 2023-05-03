import torch
import pygame.time
import flappy_bird_gym
import numpy as np
import random
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train import train_DQN
from DQN import DQN
from replaybuffer import ReplayBuffer
from evaluate import runFlappyBird,evaluateModel
from Qnet import Qnet
import time

######### 固定参数 ##########
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device=torch.device("cpu")
env = flappy_bird_gym.make('FlappyBird-v0')
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
hidden_dim = 64
buffer_size = 8000
minimal_size = 1000
target_update = 50
batch_size = 32
clock = pygame.time.Clock()
######### 固定参数 ##########

######### 评估参数设置 ##########
evaluate_episodes = 20  # 评估回合数
train_model_path="./DQN/model/model4/q_net_46500_lr_0.001_gamma_0.98.pth"
######### 评估参数设置 ##########

######### 训练参数设置 ##########
load = True            # 是否导入模型
save = True             # 是否加载模型
lr = 1e-3               # 学习率
gamma = 0.98            # 折扣因子
epsilon = 0.1           # 贪心率会动态调整
num_epoch = 1000 
num_episodes = 100000
dqn_type = "DQN"
episodes_per_epoch = num_episodes / num_epoch  # 每个epoch的episodes数
save_buffer_directory = f"./{dqn_type}/buffer/buffer1"
save_net_directory = f"./{dqn_type}/model/model1"
load_buffer_path="./DQN/buffer/buffer1/buffer_100000_lr_0.001_gamma_0.98"
load_q_net_path="./DQN/model/model1/q_net_100000_lr_0.001_gamma_0.98.pth"
load_target_net_path="./DQN/model/model1/target_net_100000_lr_0.001_gamma_0.98.pth"
######### 训练参数设置 ##########


def draw(relist,dqn_type,type="training"):
    episodes_list = list(range(len(relist)))
    plt.figure() # creat a new figure
    plt.plot(episodes_list, relist)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{dqn_type} on Flappy-Bird by {type}')
    timestrip=int(time.time())
    title=f'{dqn_type}_flappybird_{type}_{timestrip}'
    myfig = plt.gcf()
    myfig.savefig(f'./image/{title}.png')



random.seed(0)
np.random.seed(0)
env.seed(0)
torch.manual_seed(0)

def train():
    buffer = ReplayBuffer(buffer_size,load,load_buffer_path,save_buffer_directory)
    DuelingDQN_agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,
              device,load,load_q_net_path,load_target_net_path,save_net_directory,dqn_type=dqn_type)
    log_file=f'log_train_{dqn_type}_lr{lr}_gamma{gamma}_episodes{num_episodes}_targetupdate{target_update}_bats_{batch_size}.txt'
    return_list,max_q_value_list=train_DQN(DuelingDQN_agent,buffer,env,episodes_per_epoch,
                        num_epoch,minimal_size,batch_size,lr,gamma,log_file,save=save)
    draw(return_list,dqn_type)

def evaluate(render=False):
    train_agent = runFlappyBird(train_model_path, device)
    score_list,_=evaluateModel(train_agent,evaluate_episodes,render)
    draw(score_list, "","evaluating")

def getStripTime():
    return int(time.time())


if __name__ == '__main__':
    # evaluate(render=False)
    train()












