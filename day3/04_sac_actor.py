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

import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자

LOG_STD_MIN, LOG_STD_MAX = -20, 2
# 흔들림의 크기(표준편차)가 너무 작거나 너무 커지지 않게 막는 울타리입니다.
# 로그로 다루는 이유: 표준편차는 항상 양수여야 하는데,
#   신경망은 음수도 내놓습니다. 그래서 "로그값"을 내놓게 하고
#   나중에 exp() 를 씌우면 반드시 양수가 됩니다. 안전한 트릭입니다.


class GaussianActor(nn.Module):
    """
    SAC 의 배우. 어제 DDPG 배우와 결정적으로 다릅니다.

      DDPG 배우 : "힘 1.37" 이라고 딱 하나를 정한다 (결정적)
      SAC  배우 : "평균 1.37, 흔들림 0.2 정도로 뽑아라" 라고 분포를 정한다 (확률적)

    왜 분포를 내놓나요?
      탐험을 밖에서 억지로 넣지 않고 정책 자체가 갖게 하려는 것입니다.
      DDPG 는 행동에 잡음을 따로 더했는데, SAC 는 필요 없습니다.
    """

    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()                      # 부모 준비 — 빠뜨리면 오류

        # 몸통 — 상황을 이해하는 부분
        self.body = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )

        self.mu_head = nn.Linear(256, action_dim)       # 평균 (어느 쪽으로 갈지)
        self.log_std_head = nn.Linear(256, action_dim)  # 흔들림 크기의 로그
        self.max_action = max_action                    # 행동 범위 배율 (Pendulum 은 2.0)

    def forward(self, s):
        h = self.body(s)                        # 상황 이해

        mu = self.mu_head(h)                    # 평균

        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        # clamp = 울타리 안으로 자르기. 너무 작으면 -20, 너무 크면 2로.
        #   이걸 안 하면 흔들림이 0이 되거나 폭발해서 학습이 깨집니다.

        dist = torch.distributions.Normal(mu, log_std.exp())
        # 정규분포를 만든다. exp() 로 로그를 되돌리면 반드시 양수가 됩니다.

        u = dist.rsample()
        # ★ sample() 이 아니라 rsample() 입니다 ★
        #   sample()  : 그냥 뽑기. 미분이 끊깁니다.
        #   rsample() : "평균 + 흔들림 x 무작위" 형태로 뽑기. 미분이 통과합니다.
        #   미분이 통과해야 "평균을 어느 쪽으로 옮길지"를 배울 수 있습니다.
        #   이걸 재매개변수화(reparameterization) 트릭이라고 부릅니다.

        a = torch.tanh(u)                       # -1 ~ +1 사이로 누른다 (행동 범위 맞추기)

        # ── 여기가 이 코드에서 가장 어려운 두 줄입니다 ──
        log_prob = dist.log_prob(u).sum(-1)
        # 뽑은 값 u 의 확률(의 로그). sum(-1) 은 행동이 여러 개일 때 다 더하는 것.

        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1)
        # tanh 로 눌렀으니 확률도 그만큼 보정해야 합니다.
        #
        # 왜 보정이 필요한가요? (쉬운 비유)
        #   고무줄에 눈금을 그려 놓고 잡아당기면 눈금 간격이 달라집니다.
        #   tanh 는 값을 눌러 붙이는 일이라, 눌린 곳은 확률이 촘촘해집니다.
        #   그 변화만큼 빼 주는 것이 이 줄입니다.
        #   (수학에서는 '변수 변환에 따른 야코비안 보정' 이라고 부릅니다)
        #
        # + 1e-6 은 a 가 정확히 ±1 일 때 log(0) 이 되는 사고를 막는 안전장치입니다.
        #
        # ※ 이 줄을 빼먹으면 오류는 안 나는데 학습이 이상해집니다.
        #   SAC 구현에서 가장 흔한 실수입니다.

        return a * self.max_action, log_prob
        # 행동(범위 맞춘 것)과 그 행동의 로그확률을 함께 돌려준다
        # 로그확률이 필요한 이유: 엔트로피 항을 계산해야 하기 때문입니다.

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
torch.manual_seed(0)

print('확률적 배우가 잘 만들어졌는지 확인합니다.')
print()

actor = GaussianActor(state_dim=3, action_dim=1, max_action=2.0)

s = torch.randn(5, 3)                            # 상황 5개
a, logp = actor(s)

print(f'  입력 상황       {tuple(s.shape)}')
print(f'  뽑힌 행동       {tuple(a.shape)}   범위 [{a.min().item():+.3f}, {a.max().item():+.3f}]')
print('                   <- tanh x 2.0 이라 -2 ~ +2 안에 있어야 합니다')
print(f'  로그확률        {tuple(logp.shape)}')
print()

print('  같은 상황을 두 번 넣어 보면')
a1, _ = actor(s[:1])
a2, _ = actor(s[:1])
print(f'    첫 번째 {a1.item():+.4f}   두 번째 {a2.item():+.4f}')
print('    -> 값이 다릅니다. 확률적 정책이라 매번 다르게 뽑습니다.')
print('       (어제 DDPG 배우는 항상 같은 값을 냈습니다)')
print()
print('  미분이 통과하는지 (rsample 확인)')
print(f'    행동.requires_grad = {a.requires_grad}   <- True 여야 배우가 학습됩니다')
print()
print('  -> 여기까지 나오면 정상입니다. 6교시에서 이 배우로 SAC 를 돌립니다.')
