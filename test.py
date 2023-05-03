import time
import flappy_bird_gym

env = flappy_bird_gym.make("FlappyBird-v0")
count=0
score=0
while count<100:
    print(count)
    count+=1
    obs = env.reset()
    while True:
        # Next action:
        # (feed the observation to your agent here)
        # if obs[0]>0.8:
        if obs[1] > -0.0392:
            action = 0
        else:
            action = 1
        obs, reward, done, info = env.step(action)
        # print(obs)
        # Rendering the game:
        # (remove this two lines during training)
        # env.render()
        # time.sleep(1 / 50)  # FPS
        # Checking if the player is still alive
        if done:
            score+=info["score"]
            break

print(score/count)
env.close()