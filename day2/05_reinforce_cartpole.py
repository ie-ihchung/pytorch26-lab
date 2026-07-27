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

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")

policy = nn.Sequential(
    nn.Linear(4, 128), nn.ReLU(),
    nn.Linear(128, 2),          # 행동별 로짓
)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.99

for episode in range(600):
    s, _ = env.reset()
    log_probs, rewards, done = [], [], False
    while not done:                              # 1) 에피소드 수집
        logits = policy(torch.as_tensor(s, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        log_probs.append(dist.log_prob(a))
        s, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        rewards.append(r)

    G, returns = 0.0, []                         # 2) 리턴 계산 (뒤에서부터)
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # 정규화(간이 베이스라인)

    # 앞의 마이너스: 파이토치는 "줄이는" 방향으로만 움직입니다.
    # 우리는 점수를 "키우고" 싶으니 부호를 뒤집습니다.
    # 점수(returns)가 큰 행동일수록 그 행동의 확률을 크게 올립니다.
    loss = -(torch.stack(log_probs) * returns).sum()
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    if episode % 50 == 0:
        print(f"ep {episode:3d}  return {sum(rewards):.0f}")
