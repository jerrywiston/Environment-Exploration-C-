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
                            task="Exploration")
memory_size = 20000
batch_size = 64
RL = sac.SAC(
    model = {'anet':models2.PolicyNetExp2, 'qnet':models2.QNetExp2},
    n_actions = 2,
    learning_rate = [0.0001, 0.0002],
    reward_decay = 0.95,
    memory_size = memory_size,
    batch_size = batch_size,
    alpha = 0.5,
    auto_entropy_tuning=True)

#%%
is_train = False
model_path = {"actor":"models/SAC_Exp_ANet.pkl", "critic":"models/SAC_Exp_CNet.pkl"}
seq_size = 3

if not is_train:
    print("Load Model ...", model_path)
    RL.save_load_model("load", model_path)

if __name__ == '__main__':
    total_step = 0
    reward_rec= []

    for eps in range(1001):
        state = env.reset()
        state_m = cv2.resize(state["map"], (64,64), interpolation=cv2.INTER_LINEAR)
        state_m = np.tile(np.expand_dims(state_m,-1),(1,1,seq_size))
        state["map"] = state_m
        step = 0
        
        # One Episode
        eps_reward = []
        loss_a = loss_c = 0.
        while True:
            if eps>800 or not is_train:
                env.render()
            
            if is_train:
                action = RL.choose_action(state, eval=False)
            else:
                action = RL.choose_action(state, eval=True)

            # Get next state
            state_next, reward, done = env.step(action)
            reward = (reward-0) / 10
            state_m_next = cv2.resize(state_next["map"], (64,64), interpolation=cv2.INTER_LINEAR)
            state_m_next = np.concatenate([state["map"][:,:,1:seq_size], np.expand_dims(state_m_next,-1)], axis=2)
            state_next["map"] = state_m_next

            RL.store_transition(state, action, reward, state_next, done)
            eps_reward.append(reward)
            acc_reward = np.sum(np.array(eps_reward))
            if total_step > batch_size and is_train:
                loss_a, loss_c = RL.learn()

            print('\rEps: {:3d} | Step: {:4d} | Reward: {:+.3f} | Loss: [A>{:+.3f} C>{:+.3f}] | Alpha: {:.4f} | Action: [{:+.3f} {:+.3f}] | Total Reward: {:.3f}\t'\
                .format(eps, step, reward, loss_a, loss_c, RL.alpha, action[0], action[1], acc_reward), end="")
            state = state_next.copy()
            step += 1
            total_step += 1
            if done == 0. or step >= 2000:
                reward_rec.append(eps_reward)
                if eps%50 == 0 and eps > 0 and is_train:
                    print("\nSave Model ...", model_path)
                    RL.save_load_model("save", model_path)
                else:
                    print()
                break
    if is_train:
        f = open("OOO.json", "w")
        json.dump(reward_rec, f)

