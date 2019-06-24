import GSlamContBot2DWrapper
import numpy as np
import sac
import matplotlib.pyplot as plt
import json
import cv2
import models2

#%%
env = GSlamContBot2DWrapper.Bot2DEnv(obs_size=128, 
                            grid_size=3, 
                            map_path="Image/map9.png",
                            task="Navigation")
memory_size = 1000
RL = sac.SAC(
    model = {'anet':models2.PolicyNet, 'qnet':models2.QNet},
    n_actions = 2,
    learning_rate = [0.0001, 0.0002],
    reward_decay = 0.95,
    memory_size = memory_size,
    batch_size = 64,
    alpha = 0.5)

#%%
if __name__ == '__main__':
    total_step = 0
    reward_rec= []
    for eps in range(1000):
        state = env.reset()
        step = 0
        
        # One Episode
        eps_reward = []
        loss_a = loss_c = 0.
        while True:
            if eps>800:
                env.render()

            if total_step > memory_size:
                action = RL.choose_action(state)
            else:
                action = np.random.uniform(-1,1,2)

            # Get next state
            state_next, reward, done = env.step(action)
            reward = (reward-0) / 10

            RL.store_transition(state, action, reward, state_next, done)
            eps_reward.append(reward)
            if total_step > memory_size:
                loss_a, loss_c = RL.learn()

            print('\rEps: {:3d} | Step: {:3d} | Reward: {:+.3f} | Loss: [A>{:+.3f} C>{:+.3f}] | Alpha: {:.4f} | Action: [{:+.3f} {:+.3f}]\t'\
                .format(eps, step, reward, loss_a, loss_c, RL.alpha, action[0], action[1]), end="")
            state = state_next.copy()
            step += 1
            total_step += 1
            if done == 0. or step >= 600:
                reward_rec.append(eps_reward)
                print()
                break
    
    f = open("NAV2.json", "w")
    json.dump(reward_rec, f)

