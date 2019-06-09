import Nav2DWrapperMap
import numpy as np
import dqn
import matplotlib.pyplot as plt
import json
import cv2
import models
#%%
env = Nav2DWrapperMap.Bot2DEnv(obs_size=128, 
                            grid_size=3, 
                            map_path="Image/map9.png")
memory_size = 800
RL = dqn.DeepQNetwork(
                  qnet = models.QNetNavMap,
                  n_actions = 3,
                  learning_rate = 2e-4, 
                  reward_decay = 0.95,
                  replace_target_iter = 100, 
                  memory_size = memory_size,
                  batch_size = 64,
                  e_greedy = 0.95,
                  e_greedy_increment = 0.00004,)
#%%
seq_size = 3
if __name__ == '__main__':
    total_step = 0
    reward_rec= []
    for eps in range(400):
        state = env.reset()
        state_m = cv2.resize(state["map"], (64,64), interpolation=cv2.INTER_LINEAR)
        state_m = np.concatenate([np.zeros([64,64,seq_size-1], np.float32), np.expand_dims(state_m,-1)], -1)
        state["map"] = state_m
        step = 0
        
        # One Episode
        eps_reward = []
        loss = 0.
        while True:
            env.render()
            q = None
            if total_step > memory_size:
                action, q = RL.choose_action(state)
                q = np.around(q, 4)
            else:
                action = np.random.randint(0,3)

            # Get next state
            state_next, reward, done = env.step(action)
            reward = (reward-5) / 10
            state_m_next = cv2.resize(state_next["map"], (64,64), interpolation=cv2.INTER_LINEAR)
            state_m_next = np.concatenate([state["map"][:,:,1:seq_size], np.expand_dims(state_m_next,-1)], axis=2)
            state_next["map"] = state_m_next
            RL.store_transition(state, action, reward, state_next, done)
            eps_reward.append(reward)
            if total_step > memory_size:
                loss = RL.learn()

            print('\rEps: {:3d} | Step: {:3d} | Reward: {:+.4f} | Loss: {:.4f} | Q: {} | Action: {} | Epsilon: {:.5f}\t'\
                .format(eps, step, reward, loss, q, action, RL.epsilon), end="")
            state = state_next.copy()
            step += 1
            total_step += 1
            if done == 0. or step >= 600:
                reward_rec.append(eps_reward)
                print()
                break
    
    f = open("XXX.json", "w")
    json.dump(reward_rec, f)

