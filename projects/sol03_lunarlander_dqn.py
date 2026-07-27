# ==========================================================
# 미니 프로젝트 3 [응용] — LunarLander — 달 착륙선 DQN
# 환경: LunarLander-v3
# 목표: CartPole보다 어려운 8차원 상태·4행동 환경에서 DQN 계열을 튜닝한다
# 재사용: 2일차 4교시 DQN/Double DQN 코드에서 환경·네트워크 크기만 조정
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - 네트워크를 256-256으로 키우고 학습률 5e-4로 시작
#  - 바닐라 DQN으로 베이스라인 확보 (평균 리턴 기록)
#  - DOUBLE=True로 전환해 같은 조건에서 재학습, 개선 폭 확인
#  - 리플레이 버퍼 크기(1만/10만)가 성능에 주는 영향 실험
# [결과 인증]
#  최근 100 에피소드 평균 리턴 200 이상 (해결 기준) + 착륙 영상 or 학습곡선 공유

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

# pip install "gymnasium[box2d]" 필요
env = gym.make("LunarLander-v3")
obs_dim = env.observation_space.shape[0]   # 8
n_act = env.action_space.n                 # 4
DOUBLE = True                              # ← 바닐라/더블 전환 스위치

def make_net():
    return nn.Sequential(
        nn.Linear(obs_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_act),
    )

q, q_t = make_net(), make_net()
q_t.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), lr=5e-4)
buf = deque(maxlen=100_000)
gamma, batch = 0.99, 128
eps, eps_min, decay = 1.0, 0.02, 0.995

def train_step():
    s, a, r, s2, d = zip(*random.sample(buf, batch))
    s = torch.as_tensor(np.array(s), dtype=torch.float32)
    s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
    a = torch.as_tensor(a); r = torch.as_tensor(r, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)
    qv = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if DOUBLE:   # 선택=온라인넷, 평가=타깃넷
            best = q(s2).argmax(1, keepdim=True)
            nxt = q_t(s2).gather(1, best).squeeze(1)
        else:
            nxt = q_t(s2).max(1).values
        y = r + gamma * nxt * (1 - d)
    loss = nn.functional.smooth_l1_loss(qv, y)
    opt.zero_grad(); loss.backward(); opt.step()

rets = []
for ep in range(800):
    s, _ = env.reset()
    done, tot = False, 0.0
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.append((s, a, r, s2, float(term)))
        s, tot = s2, tot + r
        if len(buf) >= 2000:
            train_step()
    eps = max(eps_min, eps * decay)
    rets.append(tot)
    if ep % 10 == 0:
        q_t.load_state_dict(q.state_dict())
    if ep % 20 == 0:
        print(f"ep {ep:3d}  avg20 {np.mean(rets[-20:]):7.1f}  eps {eps:.2f}")
    if len(rets) >= 100 and np.mean(rets[-100:]) >= 200:
        print(f"해결! (ep {ep}, 최근 100ep 평균 {np.mean(rets[-100:]):.1f})")
        break
