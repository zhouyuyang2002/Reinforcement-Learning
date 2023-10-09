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

def compute_qpi_MC(pi, env, gamma, epsilon, num_episodes=1000):
    Q = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    N = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.int64)
    for i in range(num_episodes):
        state = env.reset()
        episode = []
        while True:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = pi[state]
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                episode.append((next_state, None, reward))
                break
            state = next_state
        visited = set()
        G = 0
        for j, (state, action, reward) in enumerate(reversed(episode)):
            G = gamma * G + reward
            if action == None:
                continue
            sa = (state, action)
            if sa not in visited:
                visited.add(sa)
                state = int(state)
                action = int(action)
                N[state][action] += 1
                G = gamma * G + reward
                Q[state][action] += (G - Q[state][action]) / N[state][action]
    N = np.sum(N, axis = 1)
    print(np.min(N), N)
    return Q

def test_pi(env, pi, num_episodes=100):
    
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

def policy_iteration_MC(env, gamma, eps0=0.5, decay=0.1, num_episodes=1000):

    pi = np.zeros(env.observation_space.n)
    iteration = 1
    
    import math
    nS = env.unwrapped.observation_space.n
    nA = env.unwrapped.action_space.n

    energy = 0
    while True:
        epsilon = min(eps0, eps0 / (1 + decay * iteration))
        Q = compute_qpi_MC(pi, env, gamma, epsilon, num_episodes)
        new_pi = Q.argmax(axis=1)
        energy += (pi != new_pi).sum()
        if energy <= 0:
            return new_pi, iteration
        print(f"iteration: {iteration}, eps: {epsilon}, change actions: {(pi != new_pi).sum()}, energy: {energy}")
        pi = new_pi
        iteration = iteration + 1

def test_pi(env, pi, num_episodes = 10000):
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

def test(env, num_episodes):
    sum_iter, sum_time, sum_res = 0, 0, 0
    import time
    for T in range(5):
        tic = time.time()
        pi, iter = policy_iteration_MC(env, 0.95, num_episodes = num_episodes)
        result = test_pi(env, pi)
        print("result:", result, "time:", time.time() - tic)
        sum_iter += iter
        sum_time += time.time() - tic
        sum_res += result
    
    print("average result:", sum_res / 5, "average time:", sum_time / 5, "average iter:", sum_iter / 5)


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    mdp = MDP(env)
    print("Qpi:\n", compute_qpi_MC(np.ones(16), env, gamma = 0.95, epsilon = 0.99))
    test(env, 5000)
