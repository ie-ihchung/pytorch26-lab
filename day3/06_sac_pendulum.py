# ==========================================================
# 3일차 6교시 — SAC 구현
# 2026-07-29 (수) 15:30 ~ 16:30 · Advanced Actor-Critic Methods
# 원본 파일명: sac_pendulum.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s6
# ==========================================================
# [학습목표]
#  - Pendulum-v1에서 자동 온도조절을 포함한 SAC를 완성한다
#  - DDPG 대비 학습 안정성을 비교한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import copy

env = gym.make("Pendulum-v1")
state_dim, action_dim = 3, 1
max_action = float(env.action_space.high[0])

actor = GaussianActor(state_dim, action_dim, max_action)
q1, q2 = Critic(state_dim, action_dim), Critic(state_dim, action_dim)   # 트윈 Q
q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)

# 자동 온도 조절: log_alpha를 학습, 목표 엔트로피 = -action_dim
log_alpha = torch.zeros(1, requires_grad=True)
alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
target_entropy = -action_dim

buffer = ReplayBuffer(100_000, action_dtype=torch.float32)   # 연속 행동이므로 float32
gamma, batch_size = 0.99, 256

def train_step():
    s, a, r, s_next, done = buffer.sample(batch_size)
    alpha = log_alpha.exp().detach()

    # ── 트윈 Q 업데이트: min(Q1',Q2') − α·logπ 로 soft TD 목표 ──
    with torch.no_grad():
        a_next, logp_next = actor(s_next)
        q_next = torch.min(q1_t(s_next, a_next), q2_t(s_next, a_next))
        y = r + gamma * (1 - done) * (q_next - alpha * logp_next)
    q_loss = nn.functional.mse_loss(q1(s, a), y) + nn.functional.mse_loss(q2(s, a), y)
    q_opt.zero_grad(); q_loss.backward(); q_opt.step()

    # ── Actor 업데이트: E[α·logπ − min(Q1,Q2)] 최소화 ──
    a_new, logp = actor(s)
    q_new = torch.min(q1(s, a_new), q2(s, a_new))
    actor_loss = (alpha * logp - q_new).mean()
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    # ── 온도 α 업데이트: 엔트로피를 목표치로 유지 ──
    alpha_loss = -(log_alpha.exp() * (logp + target_entropy).detach()).mean()
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    soft_update(q1_t, q1); soft_update(q2_t, q2)

returns = []
for episode in range(150):
    s, _ = env.reset()
    total, done = 0.0, False
    while not done:
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(s, dtype=torch.float32))
        s_next, r, term, trunc, _ = env.step(a.numpy())
        done = term or trunc
        buffer.push(s, a.numpy(), r, s_next, float(term))
        s, total = s_next, total + r
        if len(buffer) >= 1000:
            train_step()
    returns.append(total)
    if episode % 10 == 0:
        print(f"ep {episode:3d}  평균 {np.mean(returns[-10:]):7.1f}  alpha {log_alpha.exp().item():.3f}")
