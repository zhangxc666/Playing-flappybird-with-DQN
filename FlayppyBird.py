import random
import gym
import numpy as np
import collections
import pygame.time
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import flappy_bird_gym
import datetime
import pickle


class ReplayBuffer:
    ''' 经验回放池 '''

    def __init__(self, capacity, load=False, buffer_path=""):
        if not load:
            self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出
        else:
            try:
                self.buffer = pickle.load(open(buffer_path, 'rb'))
            except:
                print("缓冲区路径有误，请重新检查")
                exit(0)

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

    def save(self, epoch, date):
        pickle.dump(self.buffer, open(f'./buffer4/{date}_buffer_{epoch}', 'wb'))


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 128)
        self.fc2 = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(F.relu(self.fc3(x)))


class DQN:
    ''' DQN算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device,
                 load=False, train_model_path="", target_model_path=""):
        self.action_dim = action_dim
        if load:  # 是否导入模型
            try:
                self.q_net = torch.load(train_model_path)
                self.target_q_net = torch.load(target_model_path)
            except:
                print("模型路径导入错误，请重新检查")
                exit(0)
        else:
            self.q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # Q网络
            self.target_q_net = Qnet(state_dim, hidden_dim, self.action_dim).to(device)  # 目标网络

        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            if state[1] <= 0:
                action = 1
            else:
                action = 0
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)  # 下个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self, epoch, date):
        torch.save(self.q_net, f'./model4/{date}_q_net_{epoch}.pth')
        torch.save(self.target_q_net, f'./model4/{date}_target_q_net_{epoch}.pth')

    def modfiy_epsilon(self, val):
        self.epsilon = val


if __name__ == '__main__':
    ######### 加载参数 ##########
    index = 22
    target_q_net_path = f"./model2/2023-02-16_target_q_net_{index}.pth"
    q_net_path = f"./model2/2023-02-16_q_net_{index}.pth"
    buffer_path = f"./buffer2/2023-02-16_buffer_{index}"
    load = True
    save = False
    ######### 加载参数 ##########

    ######### 初始化参数 ##########
    env = flappy_bird_gym.make('FlappyBird-v0')
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    lr = 2e-3
    num_episodes = 1
    num_epoch = 1
    hidden_dim = 64
    gamma = 0.96
    epsilon = 0.03
    target_update = 50
    buffer_size = 8000
    minimal_size = 1000
    batch_size = 32
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    replaybuffer = ReplayBuffer(buffer_size, load, buffer_path)
    action_dim = env.action_space.n
    state_dim = env.observation_space.shape[0]
    today = datetime.date.today()
    rewards_list = []
    evaluate_rewards_list=[]
    evaluate_episode=10
    clock = pygame.time.Clock()
    ######### 初始化参数 ##########

    agent = DQN(state_dim, 16, action_dim, lr, gamma, epsilon, target_update, device, load, q_net_path,
                target_q_net_path)
    for epoch in range(num_epoch):
        r = 0
        with tqdm(total=int(num_episodes / num_epoch), desc='epoch %d' % epoch) as pbar:
            for i in range(int(num_episodes / num_epoch)):
                rewards = 0
                obs = env.reset()
                env.render()
                while True:
                    action = agent.take_action(obs)
                    next_obs, reward, done, _ = env.step(action)
                    env.render()
                    clock.tick(30)
                    replaybuffer.add(obs, action, reward, next_obs, done)
                    rewards += reward
                    obs = next_obs
                    if replaybuffer.size() > minimal_size and save:
                        b_s, b_a, b_r, b_ns, b_d = replaybuffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                    if done:
                        break
                rewards_list.append(rewards)
                r += rewards + 100000
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                            '%d' % (num_episodes / num_epoch * epoch + i + 1),
                        'rewards':
                            '%d' % (float(r / i + 1))
                    })
                pbar.update(1)
        # 评估模型的性能
        agent.modfiy_epsilon(0)
        from evaluate import evaluate
        evaluate_rewards_list.append(evaluate(agent,evaluate_episode))
        agent.modfiy_epsilon(epsilon)
        # 每个epoch的模型保存起来
        if save:
            agent.save(epoch, today)  # 保存agent
            replaybuffer.save(epoch, today)

    episodes_list = list(range(len(rewards_list)))
    plt.plot(episodes_list, rewards_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Flappy-Bird by training')
    plt.show()

    episodes_list=list(range(len(evaluate_rewards_list)))
    plt.plot(episodes_list, evaluate_rewards_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on Flappy-Bird by evaluating')
    plt.show()
