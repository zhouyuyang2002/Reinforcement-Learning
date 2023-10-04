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

def compute_vpi(pi, mdp, gamma, threshold = 1e-6):
    V_old = np.zeros_like(pi)
    while True:
        V_new = np.zeros_like(pi)
        for s in range(mdp.nS):
            a = pi[s]
            for p, s_, r in mdp.P[s][a]:
                V_new[s] += p * (r + gamma * V_old[s_])
        error = np.linalg.norm(V_new - V_old)
        if error < threshold:
            break
        V_old = V_new.copy()
    return V_old

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros((mdp.nS, mdp.nA))
    for s in range(mdp.nS):
        for a in range(mdp.nA):
            for p, s_, r in mdp.P[s][a]:
                Qpi[s][a] += p * (r + gamma * vpi[s_])
    return Qpi

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS, dtype = 'int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):        
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi
    return Vs, pis

def value_iteration(mdp, gamma, nIt):
    print("Iteration | max|V-Vprev| | # change actions | V[0]")
    print("----------+--------------+---------------+---------")
    Vs = [np.zeros(mdp.nS)] # 价值函数列表，初始化为0
    pis = [] # 历史策略列表
    for it in range(nIt):
        oldpi = pis[-1] if len(pis) > 0 else None 
        Vprev = Vs[-1]
        Qpi = compute_qpi(Vprev, mdp, gamma)
        pi = Qpi.argmax(axis=1)
        V = Qpi.max(axis=1)
        max_diff = np.abs(V - Vprev).max()
        nChgActions="N/A" if oldpi is None else (pi != oldpi).sum()
        print("%4i      | %6.5f      | %4s          | %5.3f"%(it, max_diff, nChgActions, V[0]))
        Vs.append(V)
        pis.append(pi)
    return Vs, pis

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

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1")
    env.reset()
    mdp = MDP(env)
    print(compute_vpi(np.ones(16), mdp, gamma = 0.95))
    print(compute_qpi(np.arange(mdp.nS), mdp, gamma = 0.95))
    """
    Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
    plt.plot(Vs_PI)
    plt.show()
    """
    Vs_PI, pis_PI = value_iteration(mdp, gamma=0.95, nIt=100)
    plt.plot(Vs_PI)
    plt.show()
    result = test_pi(env, pis_PI[-1])
    print("Prob:", result)    