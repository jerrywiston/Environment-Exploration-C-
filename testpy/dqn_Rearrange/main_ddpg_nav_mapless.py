import GSlamContBot2DWrapper
import numpy as np
import ddpg
import matplotlib.pyplot as plt
import json
import cv2
import models
from ou_noise import OUNoise
#%%
env = GSlamContBot2DWrapper.Bot2DEnv(obs_size=128, 
                            grid_size=3, 
                            map_path="Image/map9.png",
                            task="Navigation")
memory_size = 2000
RL = ddpg.DDPG(
    actor_net = models.Actor,
    critic_net = models.Critic,
    n_actions = 2,
    learning_rate = [0.0001, 0.0002],
    reward_decay = 0.95,
    memory_size = memory_size,
    batch_size = 64,
    var = 2,
    var_decay = 0.9999,)

#%%
if __name__ == '__main__':
    total_step = 0
    reward_rec= []
    exploration_noise = OUNoise(2)
    for eps in range(1000):
        state = env.reset()
        step = 0
        
        # One Episode
        eps_reward = []
        loss_a = loss_c = 0.
        while True:
            if eps>200:
                env.render()
            q = None
            if total_step > memory_size:
                action = RL.choose_action(state)
                #action = np.clip(action + exploration_noise.noise(), -1, 1)
                #action = np.clip(np.random.normal(action, 1), -1, 1)
            else:
                action = np.random.uniform(-1,1,2)

            # Get next state
            state_next, reward, done = env.step(action)
            reward = (reward-0) / 10

            RL.store_transition(state, action, reward, state_next, done)
            eps_reward.append(reward)
            if total_step > memory_size:
                loss_a, loss_c = RL.learn()

            print('\rEps: {:3d} | Step: {:3d} | Reward: {:+.3f} | Loss: [A>{:+.3f} C>{:+.3f}] | Q: {} | Action: [{:+.3f} {:+.3f}] | Var: {:.4f}\t'\
                .format(eps, step, reward, loss_a, loss_c, q, action[0], action[1], RL.var), end="")
            state = state_next.copy()
            step += 1
            total_step += 1
            if done == 0. or step >= 600:
                reward_rec.append(eps_reward)
                print()
                break
    
    f = open("OX.json", "w")
    json.dump(reward_rec, f)

