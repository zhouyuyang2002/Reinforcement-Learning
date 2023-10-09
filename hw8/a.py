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
    
def nstep_sarsa(env, num_episodes=1000, n=8, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    sum_err = 0
    num_err = 0
    sum_err_list = []
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 100 == 99:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
            sum_err_list.append(sum_err / num_err)
            sum_err = 0
            num_err = 0
        episode = []
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                episode.append((next_state, None, reward))
                break
            state = next_state
        G = 0
        episode_len = len(episode)
        td_target = [0 for _ in range(episode_len)]
        for j in reversed(range(episode_len)):
            state, action, reward = episode[j]
            G = gamma * G + reward
            td_target[j] = G
            if action == None:
                continue
            if j + n < episode_len:
                past_state, past_action, past_reward = episode[j + n]
                G = G - (gamma ** n) * past_reward
                td_target[j] = G
                if past_action != None:
                    # Not Done
                    td_target[j] = G + (gamma ** n) * Q[past_state, past_action]
            
        for j in reversed(range(episode_len)):
            state, action, reward = episode[j]
            if action != None:    
                num_err += 1
                sum_err += abs(td_target[j] - Q[state][action])
                Q[state][action] += alpha * (td_target[j] - Q[state][action])
    policy = np.argmax(Q, axis=1)
    return Q, policy, sum_err_list


def TD_lambda(env, num_episodes=1000, lmd=0.4, gamma=0.95, alpha=0.1, eps0=1, decay=0.001):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 1000 == 999:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
        episode = []
        while True:
            action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                episode.append((next_state, None, reward))
                break
            state = next_state
        episode_len = len(episode)
        for i in range(0, episode_len):
            state, action, reward = episode[i]
            if action == None:
                continue
            target_td = 0
            G = reward
            for n in range(1, episode_len - i):
                past_state, past_action, past_reward = episode[i + n]
                target_td_n = G
                if past_action != None:
                    target_td_n += (gamma ** n) * Q[past_state, past_action]
                target_td += (lmd ** (n - 1)) * target_td_n
                G = G + (gamma ** n) * past_reward
            target_td *= 1 - lmd
            target_td += (lmd ** (episode_len - i - 1)) * reward
            delta = target_td - Q[state, action]
            Q[state, action] += alpha * delta
    policy = np.argmax(Q, axis=1)
    return Q, policy

def sarsa(env, num_episodes=1000, alpha=0.1, gamma=0.99, eps0=1, decay=0.001):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    sum_err = 0
    num_err = 0
    sum_err_list = []
    for i_episode in range(num_episodes):
        state = env.reset()
        epsilon = eps0 / (1 + decay * i_episode)
        if i_episode % 100 == 99:
            print(f"iteration: {i_episode}, epsilon: {epsilon}")
            sum_err_list.append(sum_err / num_err)
            sum_err = 0
            num_err = 0

        action = epsilon_greedy(Q, state, env.action_space.n, epsilon)
        while True:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env.action_space.n, epsilon)
            td_target = reward + gamma * Q[next_state, next_action] * (not done)
            td_error = td_target - Q[state, action]
            sum_err += abs(td_error)
            num_err += 1
            Q[state, action] += alpha * td_error
            state = next_state
            action = next_action
            if done:
                break
    policy = np.argmax(Q, axis=1)
    return Q, policy, sum_err_list

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
    policy = np.argmax(Q, axis=1)
    return Q, policy

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

def test_sarsa(env):
    sum_time, sum_res = 0, 0
    for T in range(0, 5):
        import time
        tic = time.time()
        Q, pi, errs = nstep_sarsa(env, alpha = 0.01, num_episodes=10000)
        offset = time.time() - tic
        result = test_pi(env, pi)
        print("sarsa", result, "time", offset)
        sum_time += offset
        sum_res += result
    print("ave sarsa", sum_res / 5, "ave time", sum_time / 5)

    
def test_TD_lambda(env):
    sum_time, sum_res = 0, 0
    for T in range(0, 5):
        import time
        tic = time.time()
        Q, pi = TD_lambda(env, alpha = 0.1, lmd = 0.7, num_episodes=10000)
        offset = time.time() - tic
        result = test_pi(env, pi)
        print("td lambda", result, "time", offset)
        sum_time += offset
        sum_res += result
    print("ave td lambda", sum_res / 5, "ave time", sum_time / 5)

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    mdp = MDP(env)
    test_TD_lambda(env)
    """
    Q, pi = q_learning(env, num_episodes=10000)
    result = test_pi(env, pi)
    print("Q learning", result)
    """

