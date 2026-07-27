# ==========================================================
# 2일차 7교시 — A2C 구현
# 2026-07-28 (화) 16:30 ~ 17:30 · Value-based & Policy-based Methods
# 원본 파일명: a2c_cartpole.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s7
# ==========================================================
# [학습목표]
#  - CartPole-v1에서 n-step A2C를 완성한다
#  - DQN·REINFORCE와 학습 속도·안정성을 비교한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
model = ActorCritic(4, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
gamma, n_steps = 0.99, 5

s, _ = env.reset()
ep_return, returns = 0, []
for update in range(3000):
    # ── n-step 롤아웃 수집 ──
    log_probs, values, rewards, entropies, dones = [], [], [], [], []
    for _ in range(n_steps):
        logits, v = model(torch.as_tensor(s, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        s_next, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        log_probs.append(dist.log_prob(a)); values.append(v)
        rewards.append(r); dones.append(done)
        entropies.append(dist.entropy())
        ep_return += r
        s = s_next
        if done:
            returns.append(ep_return); ep_return = 0
            s, _ = env.reset()

    # ── n-step 리턴으로 TD 목표 계산 ──
    with torch.no_grad():
        _, v_last = model(torch.as_tensor(s, dtype=torch.float32))
    R, td_targets = v_last, []
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1 - d)
        td_targets.insert(0, R)
    td_targets = torch.stack(td_targets).detach()
    values = torch.stack(values)
    advantages = td_targets - values

    actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    entropy = torch.stack(entropies).mean()
    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

    optimizer.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 0.5)   # 기울기 폭주 방지
    optimizer.step()

    if update % 200 == 0 and returns:
        print(f"update {update:4d}  최근 20ep 평균 {np.mean(returns[-20:]):6.1f}")
