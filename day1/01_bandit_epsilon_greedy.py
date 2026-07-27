# ==========================================================
# 1일차 1교시 — 강화학습 소개
# 2026-07-27 (월) 09:30 ~ 10:30 · Tabular-based Methods
# 원본 파일명: bandit_epsilon_greedy.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s1
# ==========================================================
# [학습목표]
#  - 강화학습이 지도학습·비지도학습과 어떻게 다른지 설명할 수 있다
#  - 에이전트-환경 상호작용 루프(상태·행동·보상)를 이해한다
#  - 탐험(Exploration)과 활용(Exploitation)의 트레이드오프를 이해한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np

# 10개의 슬롯머신(Multi-Armed Bandit)으로 보는 탐험 vs 활용
np.random.seed(0)
n_arms = 10
true_means = np.random.normal(0, 1, n_arms)   # 각 팔의 실제 평균 보상

def run_bandit(epsilon, steps=2000):
    Q = np.zeros(n_arms)        # 행동가치 추정치
    N = np.zeros(n_arms)        # 각 팔을 당긴 횟수
    rewards = []
    for t in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(n_arms)      # 탐험
        else:
            a = np.argmax(Q)                   # 활용
        r = np.random.normal(true_means[a], 1) # 보상 샘플
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]              # 증분 평균 업데이트
        rewards.append(r)
    return np.mean(rewards)

for eps in [0.0, 0.01, 0.1, 0.5]:
    print(f"epsilon={eps:4.2f}  평균 보상 = {run_bandit(eps):.3f}")

# epsilon=0(탐험 없음)은 나쁜 팔에 갇히고,
# 0.5(과도한 탐험)는 보상을 낭비합니다. 0.1 근처가 균형점.
