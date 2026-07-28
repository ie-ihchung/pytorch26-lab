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

import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자


# ============================================================
# 오늘부터 행동이 달라집니다.
#   어제까지 : "왼쪽? 오른쪽?" — 고르는 문제 (이산)
#   오늘부터 : "힘을 1.37만큼" — 값을 정하는 문제 (연속)
#
# 왜 어제 방식이 안 통할까요?
#   어제는 모든 행동의 Q값을 내놓고 그중 max 를 골랐습니다.
#   그런데 힘의 크기는 -2.000 부터 2.000 까지 무한히 많습니다.
#   전부 계산해서 max 를 고르는 게 불가능합니다.
#   -> 그래서 "행동을 직접 내놓는 신경망"을 따로 둡니다. 그게 Actor 입니다.
# ============================================================


class Actor(nn.Module):
    """
    상황을 받아서 '할 행동'을 바로 내놓는 신경망.

    어제 정책망과 다른 점:
      어제는 확률을 내놓고 뽑았습니다 (0번 70%, 1번 30%)
      오늘은 값 하나를 딱 정합니다 (힘 1.37) — 이걸 '결정적'이라고 합니다.
    """

    def __init__(self, state_dim, action_dim, max_action):
        # state_dim  = 상황을 나타내는 숫자 개수 (Pendulum 은 3개)
        # action_dim = 행동 값이 몇 개인지 (Pendulum 은 1개 — 회전시킬 힘)
        # max_action = 행동의 최대 크기 (Pendulum 은 2.0)
        super().__init__()                      # 부모 준비 — 빠뜨리면 오류

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),      # 3개 -> 256개, 구부리기
            nn.Linear(256, 256), nn.ReLU(),            # 256 -> 256, 또 구부리기
            nn.Linear(256, action_dim), nn.Tanh(),     # 256 -> 1개, 그리고 Tanh
        )
        # 왜 마지막에 Tanh 를 붙일까요?
        #   Tanh 는 어떤 숫자가 들어와도 -1 ~ +1 사이로 눌러 줍니다.
        #   그래야 행동이 엉뚱하게 큰 값(예: 500)이 되는 것을 막습니다.
        #   어제 Q값에는 활성화를 안 붙였는데, 그건 값의 범위가 정해져
        #   있지 않았기 때문입니다. 행동은 범위가 정해져 있습니다.

        self.max_action = max_action            # 나중에 곱해 줄 배율을 기억해 둔다

    def forward(self, s):
        # Tanh 로 -1~1 이 된 값에 max_action 을 곱해 실제 범위로 늘린다
        #   예: 0.685 x 2.0 = 1.37
        return self.net(s) * self.max_action


class Critic(nn.Module):
    """
    (상황, 행동) 을 함께 받아서 "이 조합이 얼마나 좋은지" 점수를 매기는 신경망.

    ★ 어제 DQN 과 가장 크게 다른 점 ★
      어제 : 상황만 받고 -> 모든 행동의 Q값을 한꺼번에 내놓음
      오늘 : 상황 + 행동을 같이 받고 -> 그 하나에 대한 Q값만 내놓음
      행동이 무한히 많으니 "다 내놓기"가 불가능하기 때문입니다.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            #          ^^^^^^^^^^^^^^^^^^^^^ 상황과 행동을 붙여서 넣으므로 크기를 더한다
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),                  # 결과는 점수 하나
        )

    def forward(self, s, a):
        # torch.cat = 두 텐서를 옆으로 이어 붙이기
        #   상황 3개 + 행동 1개 = 4개짜리 한 줄이 됩니다.
        # dim=-1 = "맨 마지막 축 방향으로" 붙인다는 뜻
        # squeeze(-1) = (배치, 1) -> (배치,) 로 눌러 모양 맞추기
        return self.net(torch.cat([s, a], dim=-1)).squeeze(-1)


def soft_update(target, source, tau=0.005):
    """
    과녁(타깃 네트워크)을 조금씩만 따라오게 하는 함수. DDPG 와 SAC 가 함께 씁니다.

    어제와 무엇이 다른가요?
      어제 : 20판마다 과녁을 통째로 갈아 끼웠습니다 (계단처럼 툭 바뀜)
      오늘 : 매번 0.5% 씩만 섞습니다 (미끄러지듯 천천히 따라옴)

    왜 바꾸나요?
      연속 행동은 값이 아주 예민합니다. 과녁이 계단처럼 툭툭 튀면
      학습이 그때마다 흔들리다 무너집니다. 그래서 천천히 섞습니다.
    """
    # zip = 두 신경망의 숫자 뭉치를 짝지어 하나씩 꺼낸다
    for tp, sp in zip(target.parameters(), source.parameters()):
        # 새 과녁 = 0.5% 는 최신 것 + 99.5% 는 원래 과녁
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)
        # .data 를 쓰는 이유: 이건 '학습'이 아니라 '복사'입니다.
        #   미분 기록을 남기지 않고 값만 바꿔치기합니다.

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
import copy
torch.manual_seed(0)

print('배우와 평론가가 잘 만들어졌는지 확인합니다.')
print()

actor = Actor(state_dim=3, action_dim=1, max_action=2.0)   # Pendulum 규격
critic = Critic(state_dim=3, action_dim=1)

s = torch.randn(6, 3)                            # 상황 6개
a = actor(s)                                     # 배우가 행동을 내놓는다
q = critic(s, a)                                 # 평론가가 점수를 매긴다

print(f'  입력 상황        {tuple(s.shape)}')
print(f'  배우가 낸 행동   {tuple(a.shape)}   범위 [{a.min().item():+.3f}, {a.max().item():+.3f}]')
print('                    <- Tanh x 2.0 이라 -2 ~ +2 안에 있어야 합니다')
print(f'  평론가가 낸 값   {tuple(q.shape)}')
print()

# soft_update 가 실제로 조금씩만 옮기는지 확인
target = copy.deepcopy(actor)
with torch.no_grad():
    for p in actor.parameters():
        p.add_(1.0)                              # 본체를 확 바꿔 본다

before = target.net[0].weight[0, 0].item()
soft_update(target, actor, tau=0.005)
after = target.net[0].weight[0, 0].item()

print('  soft_update 확인')
print(f'    과녁 값  {before:.5f} -> {after:.5f}   (움직인 폭 {after - before:+.5f})')
print('    본체는 1.0 만큼 바뀌었는데 과녁은 0.005 만 따라왔습니다.')
print()
print('  -> 여기까지 나오면 정상입니다. 2교시에서 이 셋으로 학습을 돌립니다.')
