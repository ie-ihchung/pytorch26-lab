# ==========================================================
# 미니 프로젝트 2 [기본] — Taxi — SARSA vs Q-Learning 대결
# 환경: Taxi-v4
# 목표: 같은 문제에서 on-policy와 off-policy TD 제어의 학습 특성을 비교한다
# 재사용: 1일차 6~7교시 SARSA/Q-Learning 구현 그대로
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - Taxi-v4(상태 500개)에 두 알고리즘을 동일 하이퍼파라미터로 학습
#  - 에피소드별 리턴 곡선을 같은 그래프에 겹쳐 그리기 (matplotlib)
#  - 학습 중 곡선과 최종 greedy 정책 성능을 분리해서 비교
#  - ε을 0.3으로 키우면 두 알고리즘의 격차가 어떻게 변하는지 실험
# [결과 인증]
#  두 알고리즘 학습곡선 비교 그래프 + greedy 평가 평균 리턴 8 이상

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 그래프 한글 깨짐 방지 — 설치된 한글 폰트를 자동으로 골라 씁니다.
# Windows는 맑은 고딕, macOS는 AppleGothic이 기본 탑재입니다.
# Colab 등 리눅스는 !apt -qq install fonts-nanum 후 런타임을 다시 시작하세요.
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"):
    if _f in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

def train(method, episodes=4000, alpha=0.1, gamma=0.99, eps=0.1):
    env = gym.make("Taxi-v4")
    nS, nA = env.observation_space.n, env.action_space.n
    Q = np.zeros((nS, nA))
    rng = np.random.default_rng(0)
    returns = []

    def act(s):
        return int(rng.integers(nA)) if rng.random() < eps else int(Q[s].argmax())

    for _ in range(episodes):
        s, _ = env.reset()
        a = act(s)
        done, total = False, 0
        while not done:
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            a2 = act(s2)
            # 유일한 차이: SARSA는 실제 다음 행동 / QL은 max
            nxt = Q[s2][a2] if method == "sarsa" else Q[s2].max()
            Q[s][a] += alpha * (r + gamma * nxt * (not done) - Q[s][a])
            s, a, total = s2, a2, total + r
        returns.append(total)
    return Q, returns

def eval_greedy(Q, n=100):
    env = gym.make("Taxi-v4")
    tot = 0
    for _ in range(n):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(int(Q[s].argmax()))
            done = term or trunc
            tot += r
    return tot / n

Q_s, ret_s = train("sarsa")
Q_q, ret_q = train("qlearning")

smooth = lambda x, k=100: np.convolve(x, np.ones(k) / k, "valid")
plt.plot(smooth(ret_s), label="SARSA")
plt.plot(smooth(ret_q), label="Q-Learning")
plt.xlabel("episode"); plt.ylabel("return (100ep 이동평균)")
plt.legend(); plt.title("Taxi-v4: SARSA vs Q-Learning")
plt.show()

print(f"greedy 평가 — SARSA: {eval_greedy(Q_s):.1f} / Q-Learning: {eval_greedy(Q_q):.1f}")
# 둘 다 8 이상이면 인증! eps=0.3으로 올려 재실험해 보세요
