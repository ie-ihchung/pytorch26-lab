# ==========================================================
# 1일차 7교시 — SARSA와 Q-Learning 구현
# 2026-07-27 (월) 16:30 ~ 17:30 · Tabular-based Methods
# 원본 파일명: cliff_sarsa_qlearning.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s7
# ==========================================================
# [학습목표]
#  - Gymnasium CliffWalking 환경에서 SARSA와 Q-Learning을 완성한다
#  - 두 알고리즘이 학습한 경로의 차이를 직접 확인한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v1")
n_states, n_actions = env.observation_space.n, env.action_space.n
alpha, gamma, epsilon = 0.1, 0.99, 0.1
rng = np.random.default_rng(0)

def eps_greedy(Q, s):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    return int(np.argmax(Q[s]))

def train(method, episodes=500):
    Q = np.zeros((n_states, n_actions))
    returns = []
    for _ in range(episodes):
        s, _ = env.reset()
        a = eps_greedy(Q, s)
        total, done = 0, False
        while not done:
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if method == "sarsa":
                a_next = eps_greedy(Q, s_next)
                target = r + gamma * Q[s_next][a_next] * (not done)
            else:  # q-learning
                target = r + gamma * Q[s_next].max() * (not done)
                a_next = eps_greedy(Q, s_next)
            Q[s][a] += alpha * (target - Q[s][a])
            s, a, total = s_next, a_next, total + r
        returns.append(total)
    return Q, returns

Q_sarsa, ret_s = train("sarsa")
Q_qlearn, ret_q = train("qlearning")
print(f"SARSA      마지막 100ep 평균 보상: {np.mean(ret_s[-100:]):.1f}")
print(f"Q-Learning 마지막 100ep 평균 보상: {np.mean(ret_q[-100:]):.1f}")

# greedy 경로 시각화: SARSA는 위쪽 안전 경로, Q-Learning은 절벽 옆 최단 경로
for name, Q in [("SARSA", Q_sarsa), ("Q-Learning", Q_qlearn)]:
    grid = np.full(48, '.', dtype=str)
    s, _ = env.reset()
    for _ in range(30):
        a = int(np.argmax(Q[s]))
        grid[s] = '*'
        s, r, term, trunc, _ = env.step(a)
        if term or trunc: break
    print(f"\n[{name} greedy 경로]")
    print('\n'.join(''.join(row) for row in grid.reshape(4, 12)))
