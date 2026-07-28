# ==========================================================
# 2일차 4교시 — DQN, Double DQN 구현
# 2026-07-28 (화) 13:30 ~ 14:30 · Value-based & Policy-based Methods
# 원본 파일명: dqn_cartpole.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s4
# ==========================================================
# [학습목표]
#  - CartPole-v1에서 DQN 전체 파이프라인을 완성한다
#  - 플래그 하나로 Double DQN으로 전환해 성능을 비교한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym                        # 게임(환경)을 만들어 주는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구

# CartPole = 막대가 쓰러지지 않게 수레를 좌우로 미는 게임
# 1초 버틸 때마다 1점. 500점이 만점입니다.
env = gym.make("CartPole-v1")

obs_dim = env.observation_space.shape[0]       # 상황을 나타내는 숫자 개수 = 4
                                               # (수레 위치, 수레 속도, 막대 각도, 막대 회전속도)
n_actions = env.action_space.n                 # 할 수 있는 행동 = 2 (왼쪽 밀기, 오른쪽 밀기)

# ── 신경망을 두 개 만듭니다. 구조는 똑같고 역할만 다릅니다 ──
#   q_net    : 지금 학습하는 신경망      (화살)
#   q_target : 목표를 계산할 때만 쓰는 것 (과녁) — 한동안 고정해 둡니다
# 과녁이 계속 움직이면 맞출 수가 없습니다. 그래서 따로 둡니다.
q_net = QNetwork(obs_dim, n_actions)           # 3교시에서 만든 그 QNetwork 입니다
q_target = QNetwork(obs_dim, n_actions)
q_target.load_state_dict(q_net.state_dict())   # 처음엔 둘을 똑같이 맞춰 둔다
                                               # state_dict = 신경망 속 숫자들의 묶음

optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)   # q_net 만 학습시킨다
                                               # q_target 은 학습하지 않습니다 (복사만 받음)
buffer = ReplayBuffer(50_000)                  # 1교시에서 만든 일기장. 5만 줄까지.

gamma, batch_size = 0.99, 64
# gamma      = 미래를 얼마나 챙길지 (0.99 = 거의 다 챙긴다)
# batch_size = 한 번 배울 때 일기장에서 몇 줄을 꺼낼지

# eps = 아무 행동이나 해볼 확률 (탐험)
#   1.0 에서 시작해 매 판 0.995배씩 줄어들어 0.05 에서 멈춥니다.
#   처음엔 마구 둘러보고, 나중엔 아는 길로 갑니다.
eps, eps_min, eps_decay = 1.0, 0.05, 0.995

DOUBLE = True                                  # <- Double DQN 스위치 (False 로 바꿔 비교해 보세요)


def train_step():
    """일기장에서 조금 꺼내 한 번 배우는 함수. 위의 5단계가 그대로 들어 있습니다."""

    # 일기장에서 과거 경험 64줄을 무작위로 꺼냅니다 (섞어서 꺼내기)
    s, a, r, s_next, done = buffer.sample(batch_size)

    # q_net(s) 는 모든 행동의 값을 냅니다. 예: [3.2, 5.1]
    # 그런데 우리는 "내가 실제로 한 행동"의 값만 필요합니다 -> gather 로 뽑습니다.
    #   unsqueeze(1) : 모양을 (64,) -> (64,1) 로. gather 가 이 모양을 요구합니다.
    #   gather(1, ...) : 각 줄에서 지정한 자리 하나씩 뽑기
    #   squeeze(1)   : 뽑고 나서 (64,1) -> (64,) 로 되돌리기
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():                      # 여기부터는 '정답 만들기' — 미분 금지 구역
        if DOUBLE:
            # Double DQN: 고르는 사람과 점수 매기는 사람을 나눈다
            best_a = q_net(s_next).argmax(1, keepdim=True)          # 고르기는 학습 중인 쪽
            q_next = q_target(s_next).gather(1, best_a).squeeze(1)  # 점수는 고정된 쪽
        else:
            # 그냥 DQN: 타깃넷이 고르기도 하고 점수도 매긴다
            q_next = q_target(s_next).max(1).values

        # (1 - done) 이 오늘 가장 중요한 부분입니다.
        #   판이 끝났으면 done=1 -> 뒤쪽이 0이 되어 "미래 점수 없음"이 됩니다.
        #   이걸 빼먹으면 끝난 뒤에도 점수가 계속 더해져 값이 무한히 커집니다.
        target = r + gamma * q_next * (1 - done)

    # smooth_l1_loss = 크게 틀렸을 때 벌점을 완만하게 주는 손실
    #   강화학습은 가끔 값이 크게 튀는데, 제곱(MSE)을 쓰면 그 하나에 학습이 휘둘립니다.
    loss = nn.functional.smooth_l1_loss(q, target)

    optimizer.zero_grad()                      # ③ 지난 기울기 지우기
    loss.backward()                            # ④ 어디를 고칠지 계산
    optimizer.step()                           # ⑤ 한 걸음 이동


returns = []                                   # 판마다 받은 총점을 기록 (= 학습곡선)

for episode in range(400):                     # 400판을 한다
    s, _ = env.reset()                         # 새 판 시작. s = 지금 상황
    total, done = 0, False                     # total = 이번 판 점수

    while not done:                            # 판이 끝날 때까지 반복
        # ── 행동 고르기 (엡실론 그리디) ──
        if np.random.rand() < eps:
            a = env.action_space.sample()      # eps 확률로 아무거나 (탐험)
        else:
            with torch.no_grad():              # 행동만 고를 땐 미분 준비 불필요
                # 상황을 텐서로 바꿔 신경망에 넣고, 가장 큰 Q값의 번호를 고른다
                a = q_net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()

        # ── 그 행동을 실제로 해본다 ──
        s_next, r, term, trunc, _ = env.step(a)
        # term  = 진짜로 끝났다 (막대가 쓰러졌다)
        # trunc = 시간이 다 됐다 (500점을 채웠다)
        done = term or trunc

        # 일기장에 한 줄 적는다. float(term) 을 쓰는 이유:
        #   시간이 다 돼서 끝난 것(trunc)은 '실패'가 아닙니다.
        #   여기서 미래를 0으로 만들면 잘한 판을 나쁘게 배웁니다.
        buffer.push(s, a, r, s_next, float(term))

        s, total = s_next, total + r           # 다음 상황으로 넘어가고 점수 누적

        if len(buffer) >= 1000:                # 일기장이 1000줄 넘게 쌓인 뒤부터 배운다
            train_step()                       # (너무 적을 때 배우면 그 몇 개만 외웁니다)

    # ── 한 판이 끝난 뒤 ──
    eps = max(eps_min, eps * eps_decay)        # 탐험 확률을 조금 줄인다 (0.05 아래로는 안 감)

    # 20판마다 과녁을 최신 것으로 갈아 끼웁니다.
    #   너무 자주 하면 과녁이 흔들리고, 너무 안 하면 낡은 과녁을 보고 쏩니다.
    if episode % 20 == 0:
        q_target.load_state_dict(q_net.state_dict())

    returns.append(total)                      # 이번 판 점수 기록

    if episode % 20 == 0:                      # 20판마다 진행 상황 출력
        print(f"ep {episode:3d}  return {np.mean(returns[-20:]):6.1f}  eps {eps:.2f}")
        # 최근 20판 평균을 보는 이유: 한 판 점수는 너무 들쭉날쭉합니다.

# 최근 평균이 475를 넘으면 CartPole 을 풀었다고 봅니다.
# 400판으로는 거기까지 못 갈 수도 있습니다 — 그러면 range(400) 을 늘려 보세요.
