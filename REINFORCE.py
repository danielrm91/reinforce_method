#This method is based on the code presented in Chapter 4 from
# "Deep Reinforcement Learning in Action"

import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

#Architecture of the policy

l1 = 4
l2 = 150
l3 = 150
l4 = 2

model = torch.nn.Sequential(
    torch.nn.Linear(l1,l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2,l3),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l3,l4),
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

def discounted_rewards(rewards,gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return 

def loss_fn(preds, r):
    return -1*torch.sum(r*torch.log(preds))

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

# Training loop
episodes = 500
max_time = 200
gamma = 0.99
scores = []

for espisode in range (episodes):
    # Start the environment
    state = env.reset()[0]
    done = False
    transitions = []

    for t in range (max_time):
        acts_prob = model(torch.from_numpy(state).float())
        action = np.random.choice(np.array([0,1]), p=acts_prob.data.numpy())
        prev_state = state
        state, reward, done, info, _ = env.step(action)
        transitions.append((prev_state, action, t+1))
        if done:
            break
    ep_len = len(transitions)
    scores.append(ep_len)
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,))
    disc_rewards = discounted_rewards(reward_batch)
    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    pred_batch = model(state_batch)
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze()
    loss = loss_fn(prob_batch, disc_rewards)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(scores)
print(score.shape)
avg_score = running_mean(score, 50)
plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score, color='green')
