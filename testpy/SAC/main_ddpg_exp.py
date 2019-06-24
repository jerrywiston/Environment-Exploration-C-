import GSlamContBot2DWrapper
import numpy as np
import ddpg
import matplotlib.pyplot as plt
import json
import cv2
import models

#%%
env = GSlamContBot2DWrapper.Bot2DEnv(obs_size=128, 
                            grid_size=3, 
                            map_path="Image/map9.png",
                            task="Exploration")
memory_size = 1000
RL = ddpg.DDPG(
    actor_net = models.ActorExp,
    critic_net = models.CriticExp,
    n_actions = 2,
    learning_rate = [0.0001, 0.0002],
    reward_decay = 0.95,
    memory_size = memory_size,
    batch_size = 64,
    var = 2,
    var_decay = 0.9999,)

#%%
seq_size = 3
if __name__ == '__main__':
    total_step = 0
    reward_rec= []

    for eps in range(1000):
        state = env.reset()
        state_m = cv2.resize(state["map"], (64,64), interpolation=cv2.INTER_LINEAR)
        state_m = np.tile(np.expand_dims(state_m,-1),(1,1,seq_size))
        state["map"] = state_m
        step = 0
        
        # One Episode
        eps_reward = []
        loss_a = loss_c = 0.
        while True:
            if eps>200:
                env.render()
            if total_step > memory_size:
                action = RL.choose_action(state)
            else:
                action = np.random.uniform(-1,1,2)

            # Get next state
            state_next, reward, done = env.step(action)
            reward = (reward-0) / 10
            state_m_next = cv2.resize(state_next["map"], (64,64), interpolation=cv2.INTER_LINEAR)
            state_m_next = np.concatenate([state["map"][:,:,1:seq_size], np.expand_dims(state_m_next,-1)], axis=2)
            state_next["map"] = state_m_next

            RL.store_transition(state, action, reward, state_next, done)
            eps_reward.append(reward)
            if total_step > memory_size:
                loss_a, loss_c = RL.learn()

            print('\rEps: {:3d} | Step: {:3d} | Reward: {:+.3f} | Loss: [A>{:+.3f} C>{:+.3f}] | Action: [{:+.3f} {:+.3f}] | Var: {:.4f}\t'\
                .format(eps, step, reward, loss_a, loss_c, action[0], action[1], RL.var), end="")
            state = state_next.copy()
            step += 1
            total_step += 1
            if done == 0. or step >= 2000:
                reward_rec.append(eps_reward)
                print("\nTotal Reward: {:.4f}".format(np.sum(np.array(eps_reward))))
                break
    
    f = open("OX.json", "w")
    json.dump(reward_rec, f)

