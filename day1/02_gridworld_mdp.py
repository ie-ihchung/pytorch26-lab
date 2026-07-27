# ==========================================================
# 1일차 2교시 — MDP 소개
# 2026-07-27 (월) 10:30 ~ 11:30 · Tabular-based Methods
# 원본 파일명: gridworld_mdp.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s2
# ==========================================================
# [학습목표]
#  - 마르코프 결정 과정(MDP)의 5요소 (S, A, P, R, γ)를 설명할 수 있다
#  - 상태가치함수 V(s)와 행동가치함수 Q(s,a)의 차이를 이해한다
#  - 벨만 방정식의 재귀 구조를 이해한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np

# 4x4 GridWorld MDP 정의 — 이후 세션에서 계속 재사용합니다
# 상태: 0~15 (왼쪽 위에서 오른쪽 아래로), 0과 15는 종료 상태
# 행동: 0=상, 1=하, 2=좌, 3=우 / 보상: 이동마다 -1

N = 4
n_states = N * N
n_actions = 4
TERMINALS = [0, n_states - 1]

def step(s, a):
    """결정적 전이: (다음상태, 보상) 반환"""
    if s in TERMINALS:
        return s, 0
    r, c = divmod(s, N)
    if a == 0: r = max(r - 1, 0)
    elif a == 1: r = min(r + 1, N - 1)
    elif a == 2: c = max(c - 1, 0)
    elif a == 3: c = min(c + 1, N - 1)
    return r * N + c, -1

# 전이 텐서 P[s][a] = (s', r) 를 미리 만들어 두면 DP가 간단해집니다
P = [[step(s, a) for a in range(n_actions)] for s in range(n_states)]

# 무작위 정책의 한 에피소드 시뮬레이션
s, trajectory = 5, []
rng = np.random.default_rng(42)
while s not in TERMINALS:
    a = rng.integers(n_actions)
    s_next, r = P[s][a]
    trajectory.append((s, a, r))
    s = s_next
print(f"에피소드 길이: {len(trajectory)}, 총 보상: {sum(t[2] for t in trajectory)}")
