# ==========================================================
# 1일차 4교시 — Policy Iteration, Value Iteration 구현
# 2026-07-27 (월) 13:30 ~ 14:30 · Tabular-based Methods
# 원본 파일명: pi_vi_gridworld.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s4
# ==========================================================
# [학습목표]
#  - 4x4 GridWorld에서 정책 반복을 NumPy로 구현한다
#  - 가치 반복을 구현하고 두 방법의 수렴 결과를 비교한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np

gamma = 1.0

def q_from_v(V, s):
    """상태 s에서 각 행동의 Q값 계산"""
    return np.array([P[s][a][1] + gamma * V[P[s][a][0]]
                     for a in range(n_actions)])

# ── 정책 반복 (Policy Iteration) ──────────────────
def policy_iteration():
    policy = np.zeros(n_states, dtype=int)      # 모든 상태에서 행동 0
    while True:
        # 1) 정책 평가
        # 평가 sweep에 상한을 둡니다 (MAX_SWEEP).
        # 이유: 초기 정책(모두 '상')은 맨 윗줄에서 벽에 막혀 제자리에 머뭅니다.
        # 그러면 V[s] = -1 + 1.0 * V[s] 라서 gamma=1일 때 값이 발산해
        # delta < 1e-6 조건이 영원히 성립하지 않습니다 (무한 루프).
        # 상한을 두고 개선 단계로 넘어가면 다음 정책은 종료 상태에 도달하므로
        # 이후로는 정상 수렴합니다 — 이를 modified policy iteration이라 부릅니다.
        MAX_SWEEP = 1000
        V = np.zeros(n_states)
        for _ in range(MAX_SWEEP):
            delta = 0.0
            for s in range(n_states):
                if s in TERMINALS: continue
                s_next, r = P[s][policy[s]]
                v = r + gamma * V[s_next]
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < 1e-6: break
        # 2) 정책 개선
        stable = True
        for s in range(n_states):
            if s in TERMINALS: continue
            best_a = np.argmax(q_from_v(V, s))
            if best_a != policy[s]:
                stable = False
                policy[s] = best_a
        if stable:
            return policy, V

# ── 가치 반복 (Value Iteration) ───────────────────
def value_iteration():
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            if s in TERMINALS: continue
            v = q_from_v(V, s).max()            # max가 곧 개선
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < 1e-6: break
    policy = np.array([np.argmax(q_from_v(V, s)) for s in range(n_states)])
    return policy, V

arrows = np.array(['↑', '↓', '←', '→'])
pi_policy, pi_V = policy_iteration()
vi_policy, vi_V = value_iteration()
print("PI 최적 정책:"); print(arrows[pi_policy].reshape(4, 4))
print("VI 최적 정책:"); print(arrows[vi_policy].reshape(4, 4))
print("두 가치함수 일치:", np.allclose(pi_V, vi_V))
