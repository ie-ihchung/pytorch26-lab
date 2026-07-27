# ==========================================================
# 2일차 6교시 — Actor-Critic 소개
# 2026-07-28 (화) 15:30 ~ 16:30 · Value-based & Policy-based Methods
# 원본 파일명: actor_critic_net.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s6
# ==========================================================
# [학습목표]
#  - Actor(정책)와 Critic(가치)의 역할 분담을 이해한다
#  - 어드밴티지 함수와 A2C의 구조를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    """몸통을 공유하고 머리만 둘 — A2C의 표준 구조"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden, action_dim)   # 정책 로짓
        self.critic_head = nn.Linear(hidden, 1)           # V(s)

    def forward(self, s):
        h = self.body(s)
        return self.actor_head(h), self.critic_head(h).squeeze(-1)

# 손실 구성 미리보기 (다음 교시에 전체 루프 완성)
def a2c_loss(logits, value, action, td_target, entropy_coef=0.01):
    dist = torch.distributions.Categorical(logits=logits)
    advantage = (td_target - value).detach()      # Critic 신호는 Actor로 역전파 금지
    actor_loss = -(dist.log_prob(action) * advantage).mean()
    critic_loss = nn.functional.mse_loss(value, td_target)
    entropy = dist.entropy().mean()               # 탐험 유지 보너스
    return actor_loss + 0.5 * critic_loss - entropy_coef * entropy
