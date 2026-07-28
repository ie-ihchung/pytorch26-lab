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

import torch                                   # 파이토치

# ============================================================
# DQN 과 Double DQN 은 '정답(목표값)을 만드는 방법'만 다릅니다.
# 아래 두 함수를 나란히 두고 보면 차이가 딱 한 곳뿐인 게 보입니다.
# ============================================================


@torch.no_grad()                               # 이 함수 안에서는 미분 준비를 하지 않는다
def dqn_target(q_target, r, s_next, done, gamma=0.99):
    """
    그냥 DQN 의 목표값.
    '고르기'와 '점수 매기기'를 둘 다 타깃 네트워크가 합니다.

    왜 no_grad 인가요?
      목표값은 '과녁'입니다. 과녁은 가만히 있어야 맞출 수 있습니다.
      미분이 여기까지 흘러오면 과녁이 같이 도망갑니다.
    """
    # q_target(s_next) = 다음 상황에서 각 행동이 얼마나 좋은지 (여러 개)
    # .max(dim=1)      = 그중 가장 큰 것을 고른다 (dim=1 은 '행동 방향으로')
    # .values          = 값만 꺼낸다 (몇 번째인지는 안 쓴다)
    max_q = q_target(s_next).max(dim=1).values

    # 목표 = 지금 받은 점수 + 감마 × 다음 상황의 값
    # (1 - done) 은 "판이 끝났으면 뒤는 없다"는 뜻입니다.
    #   done=1(끝) -> 0 을 곱해 다음 값을 지운다
    #   done=0(계속) -> 1 을 곱해 그대로 둔다
    return r + gamma * max_q * (1 - done)


@torch.no_grad()
def double_dqn_target(q_online, q_target, r, s_next, done, gamma=0.99):
    """
    Double DQN 의 목표값.
    '고르는 사람'과 '점수 매기는 사람'을 나눕니다.

    왜 나누나요?
      한 사람이 고르고 그 사람이 점수까지 매기면,
      운 좋게 튄 값을 고른 뒤 그 튄 값을 그대로 믿게 됩니다.
      -> Q값이 실제보다 부풀려집니다 (최대화 편향)
    """
    # ① 고르기는 온라인 네트워크(지금 학습 중인 쪽)가 한다
    #    argmax = "가장 큰 것이 몇 번째냐" (값이 아니라 번호)
    #    keepdim=True = 모양을 (배치, 1) 로 유지 (뒤에서 gather 에 쓰려고)
    best_a = q_online(s_next).argmax(dim=1, keepdim=True)

    # ② 점수 매기기는 타깃 네트워크(잠시 고정된 쪽)가 한다
    #    gather(1, best_a) = 각 줄에서 best_a 번째 값만 골라 뽑기
    #    squeeze(1)        = (배치, 1) -> (배치,) 로 눌러서 모양 맞추기
    max_q = q_target(s_next).gather(1, best_a).squeeze(1)

    # 목표를 만드는 마지막 줄은 위와 똑같습니다.
    return r + gamma * max_q * (1 - done)


# ============================================================
# 정리 — 딱 이 차이입니다
#   DQN        : 타깃넷이 고르고, 타깃넷이 점수 매김   (혼자 다 함)
#   Double DQN : 온라인넷이 고르고, 타깃넷이 점수 매김 (역할 분담)
# ============================================================

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
import torch.nn as nn
torch.manual_seed(0)

print('두 방식이 실제로 다른 값을 내는지 확인합니다.')
print()

# 아주 작은 가짜 신경망 두 개 (온라인용, 타깃용)
q_online = nn.Linear(4, 3)                      # 상황 4개 -> 행동 3개의 Q값
q_target = nn.Linear(4, 3)

s_next = torch.randn(5, 4)                      # 다음 상황 5개
r = torch.ones(5)                               # 점수는 전부 1점
done = torch.zeros(5)                           # 아직 안 끝남

y1 = dqn_target(q_target, r, s_next, done)                    # 그냥 DQN
y2 = double_dqn_target(q_online, q_target, r, s_next, done)   # Double DQN

print('  그냥 DQN   :', [round(v, 4) for v in y1.tolist()])
print('  Double DQN :', [round(v, 4) for v in y2.tolist()])
print('  차이       :', [round(v, 4) for v in (y1 - y2).tolist()])
print()
print(f'  평균 차이  : {(y1 - y2).mean().item():+.4f}')
print('  -> 그냥 DQN 쪽이 대체로 큽니다. 이게 부풀려진 만큼입니다.')
print('     (한 번만 보면 우연일 수 있습니다. 아래 [실험] 에서 2만 번 반복해 확인합니다)')
