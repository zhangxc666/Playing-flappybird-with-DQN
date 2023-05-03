import torch
from Qnet import Qnet
import numpy as np
import torch.nn.functional as F


class DQN:
    ''' DQN算法,包括Double DQN、DQN和DuelingDQN'''

    # net1 指q_net
    # net2 指target_q_net
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update,
                 device, load=False, q_net_path="", target_net_path="", save_directory="", dqn_type=""):
        if dqn_type == "":
            print("请选择DQN算法的类型")
            exit(0)
        if dqn_type == "DuelingDQN":
            from Vnet import VAnet
            if load:
                self.q_net = torch.load(q_net_path).to(device)
                self.target_q_net = torch.load(target_net_path).to(device)
            else:
                self.q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
                self.target_q_net = VAnet(state_dim, hidden_dim, action_dim).to(device)
        elif dqn_type == "DQN" or dqn_type == "DoubleDQN":
            if load:
                self.q_net = torch.load(q_net_path, map_location='cpu').to(device)
                self.target_q_net = torch.load(target_net_path, map_location='cpu').to(device)
            else:
                self.q_net = Qnet(state_dim, hidden_dim, 128, action_dim).to(device)
                self.target_q_net = Qnet(state_dim, hidden_dim, 128, action_dim).to(device)
        else:
            print("算法类型错误")
            exit(0)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.target_update = target_update
        self.save_directory = save_directory
        self.learning_rate = learning_rate
        self.action_dim = action_dim
        self.epsilon = 1
        self.epsilon_min = 0.002
        self.decay = 0.995
        self.gamma = gamma
        self.dqn_type = dqn_type
        self.device = device
        self.count = 0

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if state[0] < 0.6:
            if np.random.random() < self.epsilon:
                action = np.random.randint(0, 2, 1, dtype=int)[0]
            else:
                state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
                action = self.q_net(state).argmax().item()
        else:
            if state[1] > -0.039:
                action = 0
            else:
                action = 1
        return action

    def max_q_value(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.int64).view(-1, 1).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(np.array(transition_dict['dones']), dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN':  # Double DQN 的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:  # DQN 和 DuelingDQN 的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self, episodes):
        torch.save(self.q_net, f'{self.save_directory}/q_net_{episodes}_lr_{self.learning_rate}_gamma_{self.gamma}.pth')
        torch.save(self.target_q_net,
                   f'{self.save_directory}/target_net_{episodes}_lr_{self.learning_rate}_gamma_{self.gamma}.pth')

    def modify_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.decay
