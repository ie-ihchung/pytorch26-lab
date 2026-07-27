# ==========================================================
# 2일차 2교시 — Double DQN 소개
# 2026-07-28 (화) 10:30 ~ 11:30 · Value-based & Policy-based Methods
# 원본 파일명: ddqn_target.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s2
# ==========================================================
# [학습목표]
#  - Q-Learning의 최대화 편향(maximization bias)이 왜 생기는지 이해한다
#  - Double DQN이 선택과 평가를 분리하는 방식을 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import torch

# DQN vs Double DQN — 목표(target) 계산의 차이
@torch.no_grad()
def dqn_target(q_target, r, s_next, done, gamma=0.99):
    max_q = q_target(s_next).max(dim=1).values          # 타깃넷이 선택+평가
    return r + gamma * max_q * (1 - done)

@torch.no_grad()
def double_dqn_target(q_online, q_target, r, s_next, done, gamma=0.99):
    best_a = q_online(s_next).argmax(dim=1, keepdim=True)   # 선택: 온라인넷
    max_q = q_target(s_next).gather(1, best_a).squeeze(1)   # 평가: 타깃넷
    return r + gamma * max_q * (1 - done)
