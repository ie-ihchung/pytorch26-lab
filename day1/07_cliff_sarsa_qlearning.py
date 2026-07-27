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

# CliffWalking = '절벽 걷기' 놀이판입니다.
# 4줄 x 12칸 격자이고, 아래쪽 가운데가 절벽입니다.
# 한 걸음마다 -1점, 절벽에 빠지면 -100점을 받고 출발점으로 돌아갑니다.
# 목표는 오른쪽 아래 끝에 도착하는 것입니다.
env = gym.make("CliffWalking-v1")

# 이 놀이판에 칸이 몇 개인지, 방향이 몇 개인지 물어봅니다.
n_states = env.observation_space.n      # 칸의 개수 (48개)
n_actions = env.action_space.n          # 방향의 개수 (4개: 상하좌우)

alpha = 0.1        # 학습률 — 새로 안 것을 10%만 반영
gamma = 0.99       # 미래를 얼마나 챙길지
epsilon = 0.1      # 아무거나 해볼 확률 — 10번에 1번

rng = np.random.default_rng(0)          # 무작위 뽑기 도구 (0은 매번 같은 결과용)


def eps_greedy(Q, s):
    """어느 방향으로 갈지 고릅니다."""
    if rng.random() < epsilon:          # 10% 확률로
        return rng.integers(n_actions)  #   → 아무 방향이나 (새로운 길 찾기)
    return int(np.argmax(Q[s]))         # 90% 는 지금까지 가장 좋았던 방향


def train(method, episodes=500):
    """500판을 하면서 표(Q)를 채워 나갑니다.

    method 가 "sarsa" 면 SARSA, 아니면 Q-러닝으로 학습합니다.
    두 방법의 차이는 아래 if 문 한 곳뿐입니다.
    """
    # 표를 0으로 채워 시작합니다. 세로 48칸 x 가로 4방향.
    Q = np.zeros((n_states, n_actions))
    returns = []                        # 판마다 받은 총점을 기록할 목록

    for _ in range(episodes):           # 500판 반복
        s, _ = env.reset()              # 출발점으로 돌아가 새 판 시작
        a = eps_greedy(Q, s)            # 첫 방향을 미리 정해 둡니다
        total = 0                       # 이번 판에서 받은 총점
        done = False                    # 판이 끝났는지

        while not done:                 # 끝날 때까지 반복
            # 정한 방향으로 한 걸음 갑니다.
            #   s_next : 가게 된 칸
            #   r      : 받은 점수 (보통 -1, 절벽이면 -100)
            #   term   : 목표에 도착했거나 절벽에 빠졌는지
            #   trunc  : 너무 오래 걸려 강제로 끊겼는지
            s_next, r, term, trunc, _ = env.step(a)
            done = term or trunc

            # ── 두 방법이 갈리는 딱 한 곳 ──────────────────
            if method == "sarsa":
                # SARSA: 다음 칸에서 "실제로 할" 방향을 먼저 정하고,
                #        그 방향의 값을 목표에 씁니다.
                #        가끔 딴 길로 새는 것까지 계산에 들어갑니다.
                a_next = eps_greedy(Q, s_next)
                target = r + gamma * Q[s_next][a_next] * (not done)
            else:
                # Q-러닝: 다음 칸에서 "가장 좋은" 방향의 값을 씁니다.
                #        실제로 그 방향으로 갈지는 상관없습니다.
                target = r + gamma * Q[s_next].max() * (not done)
                a_next = eps_greedy(Q, s_next)

            # (not done) 이 붙은 이유:
            #   판이 끝났으면 앞으로 받을 점수가 없으므로 뒤쪽을 0으로 만듭니다.
            #   이걸 빼먹으면 끝난 뒤에도 점수가 계속 더해져 숫자가 이상해집니다.

            # 표의 숫자 하나를 고칩니다. (목표 - 지금값) 만큼 조금 옮기기.
            Q[s][a] += alpha * (target - Q[s][a])

            # 다음 걸음을 위해 자리를 옮깁니다.
            s = s_next
            a = a_next
            total = total + r

        returns.append(total)           # 이번 판 총점을 기록

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
