# ==========================================================
# 3일차 1교시 — DDPG 소개
# 2026-07-29 (수) 09:30 ~ 10:30 · Advanced Actor-Critic Methods
# 원본 파일명: ddpg_networks.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s1
# ==========================================================
# [학습목표]
#  - 연속 행동 공간에서 DQN이 동작하지 않는 이유를 이해한다
#  - 결정적 정책 경사와 DDPG의 4개 네트워크 구조를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import torch
import torch.nn as nn

class Actor(nn.Module):
    """상태 → 연속 행동 (결정적)"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),   # [-1, 1]
        )
        self.max_action = max_action

    def forward(self, s):
        return self.net(s) * self.max_action         # 행동 범위로 스케일

class Critic(nn.Module):
    """(상태, 행동) → Q값 — 행동을 입력으로 받는 점이 DQN과 다름"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)

# 과녁을 조금씩만 따라오게 합니다 (tau=0.005 → 매번 0.5%씩).
# 2일차에는 20판마다 통째로 갈아 끼웠는데, 연속 행동은 값이 예민해서
# 과녁이 계단처럼 툭툭 튀면 학습이 무너집니다. 그래서 천천히 섞습니다.
def soft_update(target, source, tau=0.005):
    """타깃 네트워크 소프트 업데이트 — DDPG·SAC 공용"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
