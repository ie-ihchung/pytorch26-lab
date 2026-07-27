# ==========================================================
# 3일차 4교시 — SAC 소개
# 2026-07-29 (수) 13:30 ~ 14:30 · Advanced Actor-Critic Methods
# 원본 파일명: sac_actor.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s4
# ==========================================================
# [학습목표]
#  - SAC의 3가지 핵심 구성요소(확률적 Actor, 트윈 Q, 자동 온도조절)를 이해한다
#  - 재매개변수화 트릭이 왜 필요한지 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import torch
import torch.nn as nn

LOG_STD_MIN, LOG_STD_MAX = -20, 2

class GaussianActor(nn.Module):
    """SAC의 확률적 정책: tanh-squashed Gaussian"""
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std_head = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, s):
        h = self.body(s)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        dist = torch.distributions.Normal(mu, log_std.exp())

        u = dist.rsample()              # 재매개변수화 샘플 (기울기 통과!)
        a = torch.tanh(u)
        # tanh 변환에 따른 log-prob 보정 (change of variables)
        log_prob = dist.log_prob(u).sum(-1)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        return a * self.max_action, log_prob
