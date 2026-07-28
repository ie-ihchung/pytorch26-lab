# ==========================================================
# 2일차 5교시 — Policy Gradient 소개
# 2026-07-28 (화) 14:30 ~ 15:30 · Value-based & Policy-based Methods
# 원본 파일명: reinforce_cartpole.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s5
# ==========================================================
# [학습목표]
#  - 가치 기반과 정책 기반 접근의 차이를 이해한다
#  - REINFORCE 알고리즘과 로그-미분 트릭을 이해한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym                    # 게임(환경) 만드는 도구
import torch                               # 파이토치
import torch.nn as nn                      # 신경망 부품 상자
import numpy as np                         # 숫자 계산 도구

env = gym.make("CartPole-v1")              # 막대 세우기 게임

# ── 지금까지와 완전히 다른 접근입니다 ──
# DQN 은 "각 행동이 얼마나 좋은지(Q값)"를 배우고, 그중 큰 걸 골랐습니다.
# 여기서는 "무엇을 할지"를 바로 배웁니다. 값을 안 거칩니다.
policy = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),          # 상황 4개 -> 128개로 늘리고 구부린다
    nn.Linear(128, 2),                     # 128개 -> 2개 (행동별 '점수', logit 이라고 부름)
)
# 마지막에 softmax 를 안 붙이는 이유:
#   아래 Categorical 이 안에서 알아서 확률로 바꿔 줍니다. 두 번 하면 안 됩니다.

optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)   # 정책망을 학습시킨다
gamma = 0.99                               # 미래를 얼마나 챙길지

for episode in range(600):                 # 600판을 한다
    s, _ = env.reset()                     # 새 판 시작
    log_probs, rewards, done = [], [], False
    # log_probs = 내가 고른 행동의 '로그 확률'을 모아 둘 곳 (나중에 미분할 값)
    # rewards   = 매 걸음 받은 점수를 모아 둘 곳

    # ── 1) 한 판을 끝까지 해본다 (배우지 않고 모으기만) ──
    while not done:
        logits = policy(torch.as_tensor(s, dtype=torch.float32))   # 행동별 점수
        dist = torch.distributions.Categorical(logits=logits)      # 점수 -> 확률분포로
        a = dist.sample()                  # 확률대로 뽑는다 (항상 최선을 고르지 않음!)
                                           # 이게 곧 탐험입니다. eps 가 따로 필요 없습니다.

        log_probs.append(dist.log_prob(a)) # "그 행동을 뽑을 확률"의 로그를 저장
                                           # 왜 로그인가: 미분이 아주 간단해지기 때문 (로그미분 트릭)

        s, r, term, trunc, _ = env.step(a.item())   # 실제로 그 행동을 해본다
                                                    # a.item() = 텐서에서 숫자 하나 꺼내기
        done = term or trunc               # 쓰러졌거나 시간이 다 됐으면 끝
        rewards.append(r)                  # 받은 점수 저장

    # ── 2) 판이 끝난 뒤, 각 시점의 '앞으로 받은 총점'을 계산한다 ──
    G, returns = 0.0, []
    for r in reversed(rewards):            # 뒤에서부터 거꾸로 온다
        G = r + gamma * G                  # 지금 점수 + 감마 x 뒤에서 온 총점
        returns.insert(0, G)               # 앞쪽에 끼워 넣어 원래 순서로 되돌린다
    # 거꾸로 도는 이유: 앞에서부터 하면 매번 끝까지 다시 더해야 해서 느립니다.

    returns = torch.tensor(returns)        # 파이썬 목록 -> 텐서

    # ── 베이스라인 (오늘 배운 그것) ──
    #   평균을 빼고 표준편차로 나눕니다.
    #   빼도 되는 이유: 상태에만 의존하는 값을 빼면 평균은 그대로이고 흔들림만 줄어듭니다.
    #   안 빼면 CartPole 처럼 점수가 전부 양수인 문제에서 "다 잘했다"로 보입니다.
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    # + 1e-8 은 0으로 나누는 사고를 막는 안전장치입니다.

    # ── 3) 배우기 ──
    # 앞의 마이너스가 붙는 이유:
    #   옵티마이저는 손실을 "줄이는" 방향으로만 움직입니다.
    #   우리는 점수를 "키우고" 싶으니 부호를 뒤집어 줍니다.
    # 점수(returns)가 큰 행동일수록 그 행동의 확률을 크게 올립니다.
    loss = -(torch.stack(log_probs) * returns).sum()
    # torch.stack = 낱개 텐서들을 하나로 쌓기

    optimizer.zero_grad()                  # ③ 지난 기울기 지우기
    loss.backward()                        # ④ 어디를 고칠지 계산
    optimizer.step()                       # ⑤ 한 걸음 이동

    if episode % 50 == 0:                  # 50판마다 출력
        print(f"ep {episode:3d}  return {sum(rewards):.0f}")

# DQN 과 비교해 보세요.
#   DQN     : 매 걸음마다 조금씩 배운다 (일기장에서 꺼내서)
#   REINFORCE : 한 판이 끝나야 배운다 (끝까지 가봐야 총점을 알 수 있으니까)
#   -> 그래서 REINFORCE 는 배우는 횟수가 적고, 결과가 더 들쭉날쭉합니다.
#      이 문제를 6교시 Actor-Critic 이 해결합니다.
