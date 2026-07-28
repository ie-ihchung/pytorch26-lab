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

import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자


class ActorCritic(nn.Module):
    """
    배우(Actor) 와 평론가(Critic) 를 한 몸에 넣은 신경망.

    배우   : 무엇을 할지 정한다
    평론가 : 지금 상황이 얼마나 좋은지 점수를 매긴다

    왜 한 몸인가요?
      둘 다 "지금 상황을 이해하는 일"은 똑같이 필요합니다.
      그 공통 부분(몸통)을 함께 쓰고, 마지막 판단만 나눕니다.
      사람으로 치면 눈은 하나인데 판단하는 머리만 둘인 셈입니다.
    """

    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()                      # 부모 준비 — 빠뜨리면 오류

        # 몸통 — 상황을 이해하는 부분. 배우와 평론가가 함께 씁니다.
        self.body = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),   # 4개 -> 128개, 구부리기
        )

        self.actor_head = nn.Linear(hidden, action_dim)  # 배우 머리: 행동별 점수(logit)
        self.critic_head = nn.Linear(hidden, 1)          # 평론가 머리: 값 하나 V(s)

    def forward(self, s):
        h = self.body(s)                        # 먼저 몸통을 통과 (상황 이해)
        return (
            self.actor_head(h),                 # 무엇을 할지 (여러 개)
            self.critic_head(h).squeeze(-1),    # 얼마나 좋은지 (하나)
        )
        # squeeze(-1) = (배치, 1) -> (배치,) 로 눌러서 모양 맞추기
        # 안 하면 나중에 계산할 때 모양이 안 맞아 조용히 잘못된 값이 나옵니다.


def a2c_loss(logits, value, action, td_target, entropy_coef=0.01):
    """
    A2C 의 손실. 세 조각을 더한 것입니다.
    (전체 학습 루프는 다음 7교시에 완성합니다)
    """

    dist = torch.distributions.Categorical(logits=logits)   # 점수 -> 확률분포로

    # ── 어드밴티지 = "예상보다 얼마나 좋았나" ──
    #   td_target : 실제로 겪어 보니 이 정도였다
    #   value     : 평론가가 예상했던 값
    #   차이가 +면 예상보다 좋았다는 뜻 -> 그 행동의 확률을 올린다
    advantage = (td_target - value).detach()
    # ★ .detach() 를 빠뜨리면 안 됩니다 ★
    #   이걸 안 붙이면 "배우를 고치려던 신호"가 평론가까지 흘러갑니다.
    #   평론가는 자기 손실(아래 critic_loss)로만 배워야 합니다.
    #   섞이면 평론가가 "값을 정확히 맞히기"가 아니라
    #   "배우가 좋아할 값 내놓기"를 배워 버립니다.

    # ── 조각 ① 배우 손실 ──
    # 어드밴티지가 큰 행동일수록 그 행동의 확률을 크게 올린다
    # 앞의 마이너스는 "키우고 싶다"를 "줄이고 싶다"로 뒤집은 것
    actor_loss = -(dist.log_prob(action) * advantage).mean()

    # ── 조각 ② 평론가 손실 ──
    # 평론가의 예상(value)이 실제(td_target)에 가까워지게
    critic_loss = nn.functional.mse_loss(value, td_target)

    # ── 조각 ③ 엔트로피 (탐험 유지) ──
    # 엔트로피 = 얼마나 헷갈려 하는가.
    #   (0.5, 0.5) 면 크고, (0.99, 0.01) 이면 작습니다.
    entropy = dist.entropy().mean()

    # 세 조각을 더합니다. 계수는 셋의 크기를 맞추는 저울입니다.
    #   0.5  : 평론가 손실은 값이 크게 나오기 쉬워 절반으로 눌러 줍니다.
    #          안 그러면 배우 손실이 묻혀 버립니다.
    #   -0.01 : 엔트로피는 빼 줍니다. 헷갈리는 쪽이 손실이 작아지므로
    #           너무 일찍 한 행동만 고집하지 않게 붙잡아 줍니다.
    return actor_loss + 0.5 * critic_loss - entropy_coef * entropy

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
torch.manual_seed(0)

print('배우와 평론가가 잘 만들어졌는지 확인합니다.')
print()

model = ActorCritic(state_dim=4, action_dim=2)   # 상황 4개, 행동 2개

s = torch.randn(8, 4)                            # 상황 8개를 한 번에 (배치)
logits, value = model(s)

print(f'  입력 모양            {tuple(s.shape)}')
print(f'  배우가 낸 것(logits) {tuple(logits.shape)}   <- 상황마다 행동 2개의 점수')
print(f'  평론가가 낸 것(V)    {tuple(value.shape)}      <- 상황마다 값 하나')
print()

# 손실이 계산되는지 확인
action = torch.randint(0, 2, (8,))               # 실제로 한 행동
td_target = torch.randn(8)                       # 목표값 (가짜)
loss = a2c_loss(logits, value, action, td_target)

print(f'  a2c_loss 계산 결과   {loss.item():.4f}')
print(f'  미분 연결 상태       {loss.requires_grad}  <- True 여야 학습이 됩니다')
print()
print('  -> 여기까지 나오면 정상입니다. 7교시에서 이 조각들로 학습 루프를 완성합니다.')
