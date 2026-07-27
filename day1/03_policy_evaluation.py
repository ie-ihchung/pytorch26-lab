# ==========================================================
# 1일차 3교시 — Dynamic Programming 소개
# 2026-07-27 (월) 11:30 ~ 12:30 · Tabular-based Methods
# 원본 파일명: policy_evaluation.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s3
# ==========================================================
# [학습목표]
#  - 정책 평가(Policy Evaluation)와 정책 개선(Policy Improvement)을 구분한다
#  - 정책 반복(PI)과 가치 반복(VI)의 차이를 설명할 수 있다
#  - DP가 model-based 방법인 이유와 한계를 이해한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np

# 무작위 정책에 대한 반복적 정책 평가 (4x4 GridWorld, 앞 세션의 P 재사용)
gamma = 1.0
theta = 1e-6          # 수렴 판정 기준

V = np.zeros(n_states)
iteration = 0
while True:
    delta = 0.0
    V_new = V.copy()
    for s in range(n_states):
        if s in TERMINALS:
            continue
        # 무작위 정책: 4방향 각 0.25 확률
        v = 0.0
        for a in range(n_actions):
            s_next, r = P[s][a]
            v += 0.25 * (r + gamma * V[s_next])
        V_new[s] = v
        delta = max(delta, abs(v - V[s]))
    V = V_new
    iteration += 1
    if delta < theta:
        break

print(f"{iteration}회 반복 후 수렴")
print(np.round(V.reshape(4, 4), 1))
# 종료 상태에서 멀수록 가치가 낮아지는(-22 근처) 것을 확인하세요
