# ==========================================================
# 미니 프로젝트 7 [심화] — MountainCar — Sparse Reward 정복
# 환경: MountainCar-v0
# 목표: 보상이 극도로 희소한 환경에서 탐험 전략의 한계와 해법을 체험한다
# 재사용: 2일차 4교시 DQN 코드에서 시작
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - 기본 DQN이 왜 실패하는지 확인 (정상 도달 전까지 보상 신호 전무)
#  - 1단계 해법 — 보상 성형: 위치·속도 기반 보조 보상 추가 후 재학습
#  - 2단계 해법 — 간이 curiosity: 다음 상태 예측 모델의 예측 오차를 내적 보상으로 추가
#  - 두 해법의 학습 속도와 최종 정책 품질 비교
# [결과 인증]
#  정상 도달 성공(평가 10회 중 8회 이상) + 사용한 해법 설명 공유

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

# MODE: "none"(실패 확인용) / "shaping"(보상 성형) / "curiosity"(예측오차 보상)
MODE = "shaping"

env = gym.make("MountainCar-v0")
obs_dim, n_act = 2, 3

def make_net():
    return nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                         nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, n_act))

q, q_t = make_net(), make_net()
q_t.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), lr=1e-3)
buf = deque(maxlen=50_000)
gamma, batch, eps = 0.99, 64, 1.0

# 간이 curiosity: 다음 상태를 예측하는 forward 모델 (예측이 틀릴수록 보상)
fwd = nn.Sequential(nn.Linear(obs_dim + 1, 64), nn.ReLU(), nn.Linear(64, obs_dim))
fwd_opt = torch.optim.Adam(fwd.parameters(), lr=1e-3)

def bonus(s, a, s2):
    if MODE == "shaping":
        # 속도 기반 potential shaping — 빨라질수록 보너스 (최적 정책 보존형)
        return 300.0 * (gamma * abs(s2[1]) - abs(s[1]))
    if MODE == "curiosity":
        x = torch.as_tensor(np.append(s, a / 2.0), dtype=torch.float32)
        pred = fwd(x)
        target = torch.as_tensor(s2, dtype=torch.float32)
        err = nn.functional.mse_loss(pred, target)
        fwd_opt.zero_grad(); err.backward(); fwd_opt.step()   # 예측모델도 학습
        return 100.0 * float(err.detach())                     # 놀라움 = 내적 보상
    return 0.0

def train_step():
    s, a, r, s2, d = zip(*random.sample(buf, batch))
    s = torch.as_tensor(np.array(s), dtype=torch.float32)
    s2 = torch.as_tensor(np.array(s2), dtype=torch.float32)
    a = torch.as_tensor(a); r = torch.as_tensor(r, dtype=torch.float32)
    d = torch.as_tensor(d, dtype=torch.float32)
    qv = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        y = r + gamma * q_t(s2).max(1).values * (1 - d)
    loss = nn.functional.smooth_l1_loss(qv, y)
    opt.zero_grad(); loss.backward(); opt.step()

success = 0
for ep in range(500):
    s, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buf.append((s, a, r + bonus(s, a, s2), s2, float(term)))
        s = s2
        if len(buf) >= 1000:
            train_step()
    eps = max(0.05, eps * 0.995)
    if ep % 10 == 0:
        q_t.load_state_dict(q.state_dict())
    if term and s[0] >= 0.5:
        success += 1
        if success == 1:
            print(f"첫 정상 도달! ep {ep}")

# ── 평가 10회 ──
wins = 0
for _ in range(10):
    s, _ = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            a = int(q(torch.as_tensor(s, dtype=torch.float32)).argmax())
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
    wins += (term and s[0] >= 0.5)
print(f"평가 성공 {wins}/10   (8 이상 인증!)  MODE={MODE}")
# MODE="none"으로 바꿔 다시 돌리면 왜 실패하는지 체감할 수 있습니다
