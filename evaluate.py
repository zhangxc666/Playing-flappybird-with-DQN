import pygame.time
from tqdm import tqdm
import rl_utils
import torch
import flappy_bird_gym
import matplotlib.pyplot as plt
from Qnet import Qnet
import pygame
clock = pygame.time.Clock()
class runFlappyBird:
    def __init__(self, path, device):
        self.device = device
        self.q_net = torch.load(path).to(device)

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.q_net(state).argmax().item()
        return action


def evaluateModel(agent,evaluate_episodes,render=False,count=1):
    env = flappy_bird_gym.make('FlappyBird-v0')
    env.seed(0)
    scores=0
    score_list=[]
    with tqdm(range(evaluate_episodes), desc="model_evaluate",colour='blue') as tbar:
        for j in range(evaluate_episodes):
            state = env.reset()
            if render:
                env.render()
            while True:
                action = agent.take_action(state)
                next_state, reward, done, _ = env.step(action)
                if render:
                    env.render()
                    clock.tick(300)
                state = next_state
                if done:
                    score=_["score"]
                    score_list.append(score)
                    break
            scores+=score
            tbar.update(1)
            tbar.set_postfix({'count':count,'mean_score': scores / (j + 1)})
    return score_list,scores/evaluate_episodes


if __name__ == '__main__':
    device=torch.device("cpu")
    lr=0.001
    gamma=0.98
    targetupdate=50
    batch_size=32
    dqn_type="DoubleDQN"
    avg_score_list=[]
    log_file=f"avgscore_{dqn_type}_lr{lr}_gamma{gamma}_episodes{100000}_targetupdate{targetupdate}_bats{batch_size}"
    for episode in range(50,100001,50):
        path=f"./{dqn_type}/model/model4/q_net_{episode}_lr_{lr}_gamma_{gamma}.pth"
        agent=runFlappyBird(path,device)
        _,avg_score=evaluateModel(agent,25,False,episode)
        avg_score_list.append(avg_score)
    with open(f'./avg_log/{log_file}','a') as f:
        for i in avg_score_list:
            f.write(f'{i}\n') 
    from main import draw
    draw(avg_score_list,dqn_type,"evaluate")