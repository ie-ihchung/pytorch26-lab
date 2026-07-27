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

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]     # 4
n_actions = env.action_space.n               # 2

# 신경망을 두 개 만듭니다. 구조는 똑같고 역할만 다릅니다.
#  q_net    : 지금 학습하는 신경망 (화살)
#  q_target : 목표 계산 전용, 한동안 고정 (과녁)
# 처음에는 둘을 똑같이 맞춰 둡니다.
q_net = QNetwork(obs_dim, n_actions)
q_target = QNetwork(obs_dim, n_actions)
q_target.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer(50_000)

gamma, batch_size = 0.99, 64
# eps = 아무 행동이나 해볼 확률.
#   1.0에서 시작해 매 판 0.995배씩 줄어들어 0.05에서 멈춥니다.
#   처음엔 마구 둘러보고, 나중엔 아는 길로 가는 것입니다.
eps, eps_min, eps_decay = 1.0, 0.05, 0.995
DOUBLE = True                                # ← Double DQN 스위치

def train_step():
    # 상자에서 과거 경험 64개를 무작위로 꺼냅니다.
    s, a, r, s_next, done = buffer.sample(batch_size)

    # q_net(s) 는 모든 행동의 값을 냅니다. 예: [3.2, 5.1]
    # 그중 "내가 실제로 한 행동"의 값만 뽑아야 합니다 → gather
    #   unsqueeze(1) : 모양을 (64,) → (64,1) 로. gather 가 요구하는 형태입니다.
    #   squeeze(1)   : 뽑고 나서 (64,1) → (64,) 로 되돌립니다.
    q = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if DOUBLE:   # 선택은 온라인넷, 평가는 타깃넷
            best_a = q_net(s_next).argmax(1, keepdim=True)
            q_next = q_target(s_next).gather(1, best_a).squeeze(1)
        else:        # 타깃넷이 선택+평가 (바닐라 DQN)
            q_next = q_target(s_next).max(1).values
        # (1 - done) 이 오늘 가장 중요한 부분입니다.
        # 게임이 끝났다면 done=1 → 뒤쪽이 0이 되어 "미래 점수 없음"이 됩니다.
        # 이걸 빼먹으면 끝난 뒤에도 점수가 계속 더해져 값이 무한히 커집니다.
        target = r + gamma * q_next * (1 - done)

    # smooth_l1_loss : 오차가 클 때 덜 민감한 손실.
    #   강화학습은 가끔 튀는 값이 있어서 이 손실이 더 안정적입니다.
    loss = nn.functional.smooth_l1_loss(q, target)
    optimizer.zero_grad()   # 이전 기울기 지우기
    loss.backward()         # 기울기 계산
    optimizer.step()        # 가중치 갱신

returns = []
for episode in range(400):
    s, _ = env.reset()
    total, done = 0, False
    while not done:
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q_net(torch.as_tensor(s, dtype=torch.float32)).argmax().item()
        s_next, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.push(s, a, r, s_next, float(term))
        s, total = s_next, total + r
        if len(buffer) >= 1000:
            train_step()
    eps = max(eps_min, eps * eps_decay)
    # 20판마다 과녁을 최신 것으로 갈아 끼웁니다.
    # 너무 자주 하면 과녁이 흔들리고, 너무 안 하면 낡은 과녁을 보고 쏩니다.
    if episode % 20 == 0:
        q_target.load_state_dict(q_net.state_dict())
    returns.append(total)
    if episode % 20 == 0:
        print(f"ep {episode:3d}  return {np.mean(returns[-20:]):6.1f}  eps {eps:.2f}")
# 평균 리턴이 475를 넘으면 CartPole 해결!
