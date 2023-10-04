import gym
import numpy as np
import matplotlib.pyplot as plt
def test_env():
    env = gym.make("FrozenLake-v1")
    env.reset()
    for t in range(100):
        env.render()
        a = env.action_space.sample()
        observation, reward, done, _, info = env.step(a)
        print(a, observation, reward, done, _, info)
        if done:
            break
    env.render()
    while True:
        continue

def test_trans():
    env = gym.make("FrozenLake-v1")
    env.reset()
    print(env.unwrapped.P)

class MDP(object):
    def __init__(self, env):
        env = env.unwrapped
        self.P = {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}
        self.nS = env.unwrapped.observation_space.n
        self.nA = env.unwrapped.action_space.n

def epsilon_greedy(Q, state, num_actions, epsilon):
    if np.random.random() < epsilon:
        return np.random.choice(num_actions)
    else:
        return np.argmax(Q[state, :])
    
def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        print(f"iteration: {i_episode}, epsilon: {epsilon}")
        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target-Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            action = next_action
            if done:
                break
    policy = np.argmax(Q, axis=1)
    return Q, policy

def q_learning(env, num_episodes=1000, alpha=0.05, gamma=0.99, eps0=1, decay=0.001):
    # 初始化Q表为0
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    # 针对每个回合进行更新
    for i_episode in range(num_episodes):
        # 初始化状态
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            if done:
                break
        if i_episode % 10000 == 0:
            print(Q)
            print(epsilon)
    policy = np.argmax(Q, axis=1)
    return Q, policy

def test_pi(env, pi, num_episodes = 100):
    count = 0
    for e in range(num_episodes):
        ob = env.reset()
        for t in range(100):
            a = pi[ob]
            ob, rew, done, _ = env.step(a)
            if done:
                count += 1 if rew == 1 else 0
                break
    return count / num_episodes

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    mdp = MDP(env)
    Q, pi = sarsa(env, num_episodes=10000)
    result = test_pi(env, pi)
    print("sarsa", result)
    Q, pi = q_learning(env, num_episodes=10000)
    result = test_pi(env, pi)
    print("Q learning", result)


