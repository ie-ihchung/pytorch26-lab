# ==========================================================
# 3일차 2교시 — DDPG 구현
# 2026-07-29 (수) 10:30 ~ 11:30 · Advanced Actor-Critic Methods
# 원본 파일명: ddpg_pendulum.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s2
# ==========================================================
# [학습목표]
#  - Pendulum-v1(연속 행동)에서 DDPG 학습 루프를 완성한다
#  - 소프트 업데이트와 탐험 노이즈의 효과를 확인한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import copy

env = gym.make("Pendulum-v1")
state_dim = env.observation_space.shape[0]   # 3
action_dim = env.action_space.shape[0]       # 1
max_action = float(env.action_space.high[0]) # 2.0

actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)
actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
buffer = ReplayBuffer(100_000, action_dtype=torch.float32)   # 연속 행동이므로 float32
gamma, batch_size, noise_std = 0.99, 128, 0.1

def train_step():
    s, a, r, s_next, done = buffer.sample(batch_size)
    # ── Critic 업데이트: TD 목표는 타깃 네트워크들로 ──
    with torch.no_grad():
        target_q = critic_t(s_next, actor_t(s_next))
        y = r + gamma * target_q * (1 - done)
    critic_loss = nn.functional.mse_loss(critic(s, a), y)
    critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

    # ── Actor 업데이트: Q(s, μ(s))를 최대화 ──
    actor_loss = -critic(s, actor(s)).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    soft_update(actor_t, actor); soft_update(critic_t, critic)

returns = []
for episode in range(200):
    s, _ = env.reset()
    total, done = 0.0, False
    while not done:
        with torch.no_grad():
            a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
        # 배우는 항상 같은 답을 냅니다. 그대로 두면 새로운 걸 안 해봅니다.
        # 그래서 답에 살짝 흔들림을 더합니다. "32도" 대신 "32.4도" 처럼.
        a += np.random.normal(0, noise_std * max_action, action_dim)
        a = a.clip(-max_action, max_action)
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.push(s, a, r, s_next, float(term))
        s, total = s_next, total + r
        if len(buffer) >= 1000:
            train_step()
    returns.append(total)
    if episode % 10 == 0:
        print(f"ep {episode:3d}  최근 10ep 평균 리턴 {np.mean(returns[-10:]):7.1f}")
# -1500 근처에서 시작해 -200 근처까지 오르면 성공
