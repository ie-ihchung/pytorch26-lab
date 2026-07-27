# ==========================================================
# 1일차 6교시 — SARSA와 Q-Learning 소개
# 2026-07-27 (월) 15:30 ~ 16:30 · Tabular-based Methods
# 원본 파일명: update_rules.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s6
# ==========================================================
# [학습목표]
#  - TD 제어에서 SARSA와 Q-Learning의 업데이트 식을 구분한다
#  - On-policy와 Off-policy의 차이를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

# 두 알고리즘의 차이는 단 한 줄 — TD 목표(target)의 정의

# SARSA (on-policy): 다음 "실제" 행동 a'의 Q값 사용
def sarsa_update(Q, s, a, r, s_next, a_next, alpha=0.1, gamma=0.99):
    target = r + gamma * Q[s_next][a_next]        # 실제 선택한 a'
    Q[s][a] += alpha * (target - Q[s][a])

# Q-Learning (off-policy): 다음 상태의 "최대" Q값 사용
def q_learning_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    target = r + gamma * max(Q[s_next])           # max — 실제 행동과 무관
    Q[s][a] += alpha * (target - Q[s][a])

# 행동 선택은 둘 다 epsilon-greedy
import random
def epsilon_greedy(Q, s, n_actions, epsilon=0.1):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return max(range(n_actions), key=lambda a: Q[s][a])
