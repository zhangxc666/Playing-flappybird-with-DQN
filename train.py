from tqdm import tqdm
import pygame
clock = pygame.time.Clock()
def train_DQN(agent,replay_buffer,env,episodes_per_epoch,num_epoch,
              minimal_size,batch_size,learning_rate,gamma,log_file,save=True):
    score_list = []
    max_q_value_list = []
    max_q_value = 0
    for epoch in range(num_epoch):
        score_epoch=0
        with tqdm(total=int(episodes_per_epoch), desc='Iteration %d' % epoch) as pbar:
            for episode in range(int(episodes_per_epoch)):
                obs = env.reset()
                rewards=0
                while True:
                    action = agent.take_action(obs)
                    max_q_value = agent.max_q_value(obs) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    next_obs, reward, done, _ = env.step(action)
                    rewards+=reward
                    if obs[0]<0.6:
                        replay_buffer.add(obs, action, reward, next_obs, done)
                    obs = next_obs
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,'dones': b_d}
                        agent.update(transition_dict)
                    if done:
                        score_episode=_["score"]
                        break
                score_epoch+=score_episode
                score_list.append(score_episode)
                if (episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (episodes_per_epoch * epoch + episode + 1),
                        'avg_scores': '%d' % (float(score_epoch / (episode + 1))),
                        'sums':'%d'%(score_epoch),
                    })
                pbar.update(1)
                agent.modify_epsilon()
                with open(f'./log/log2_best/{log_file}','a') as f:
                    f.write(f'{score_episode},{rewards}\n')
        if save:
            agent.save(int((epoch+1)*episodes_per_epoch))
            replay_buffer.save(learning_rate,int((epoch+1)*episodes_per_epoch),gamma)
    return score_list, max_q_value_list
