# ==========================================================
# 1일차 5교시 — Monte-Carlo 방법, Temporal Difference 방법 소개
# 2026-07-27 (월) 14:30 ~ 15:30 · Tabular-based Methods
# 원본 파일명: mc_vs_td_prediction.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s5
# ==========================================================
# [학습목표]
#  - 모델 없이(model-free) 가치를 추정하는 두 접근을 이해한다
#  - MC의 낮은 편향·높은 분산, TD의 부트스트래핑·낮은 분산 특성을 비교한다
#  - TD(0) 업데이트 식을 쓸 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np

# 무작위 정책의 V를 MC와 TD(0)로 각각 추정해 비교 (4x4 GridWorld)
rng = np.random.default_rng(0)
alpha, gamma = 0.05, 1.0

def gen_episode():
    s = rng.integers(1, n_states - 1)
    episode = []
    while s not in TERMINALS:
        a = rng.integers(n_actions)
        s_next, r = P[s][a]
        episode.append((s, r))
        s = s_next
    return episode

# ── First-visit Monte-Carlo ──
V_mc = np.zeros(n_states)
for _ in range(5000):
    episode = gen_episode()
    G, visited = 0.0, set()
    for s, r in reversed(episode):     # 뒤에서부터 리턴 누적
        G = r + gamma * G
        if s not in visited:           # first-visit
            visited.add(s)
            V_mc[s] += alpha * (G - V_mc[s])

# ── TD(0) ──
V_td = np.zeros(n_states)
for _ in range(5000):
    s = rng.integers(1, n_states - 1)
    while s not in TERMINALS:
        a = rng.integers(n_actions)
        s_next, r = P[s][a]
        td_error = r + gamma * V_td[s_next] - V_td[s]
        V_td[s] += alpha * td_error
        s = s_next

print("MC 추정:"); print(np.round(V_mc.reshape(4, 4), 1))
print("TD 추정:"); print(np.round(V_td.reshape(4, 4), 1))
# 둘 다 DP 정답(-14, -20, -22...)에 근접하는지 확인하세요
