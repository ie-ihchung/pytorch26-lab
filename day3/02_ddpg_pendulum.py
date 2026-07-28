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

import gymnasium as gym                        # 게임(환경) 만드는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구
import copy                                    # 통째로 복사할 때 쓰는 도구

# Pendulum = 막대를 흔들어서 거꾸로 세우는 게임
#   점수가 항상 음수입니다. 0에 가까울수록 잘한 것입니다.
#   아무렇게나 하면 -1200 근처가 나옵니다.
env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]     # 상황을 나타내는 숫자 = 3개
                                               # (막대 각도의 cos, sin, 회전 속도)
action_dim = env.action_space.shape[0]         # 행동 값 = 1개 (돌리는 힘)
max_action = float(env.action_space.high[0])   # 힘의 최대 크기 = 2.0

# ── 신경망 4개를 만듭니다 ──
#   actor    : 무엇을 할지 정하는 쪽 (배우)
#   critic   : 그게 얼마나 좋은지 점수 매기는 쪽 (평론가)
#   actor_t  : 배우의 과녁 (천천히 따라오는 복사본)
#   critic_t : 평론가의 과녁
actor = Actor(state_dim, action_dim, max_action)
critic = Critic(state_dim, action_dim)

actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
# deepcopy = 완전히 똑같은 별개의 복사본을 만든다
#   그냥 actor_t = actor 라고 하면 같은 것을 가리켜서 과녁 역할을 못 합니다.

actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)    # 배우용 옵티마이저
critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)  # 평론가용 옵티마이저
# 배우의 학습률이 10배 작습니다.
#   평론가가 아직 엉터리인데 배우가 빨리 따라가면 엉뚱한 걸 배웁니다.
#   "평론가가 먼저 실력을 갖추고, 배우는 천천히" 라는 뜻입니다.

buffer = ReplayBuffer(100_000, action_dtype=torch.float32)
# ★ action_dtype=torch.float32 를 꼭 넣어야 합니다 ★
#   행동이 1.37 같은 실수인데 정수로 두면 1 로 잘려 학습이 통째로 망가집니다.

gamma, batch_size, noise_std = 0.99, 128, 0.1
# gamma      = 미래를 얼마나 챙길지
# batch_size = 한 번에 128개씩 꺼내 배운다
# noise_std  = 행동에 섞을 흔들림의 크기 (아래에서 설명)


def train_step():
    """일기장에서 꺼내 한 번 배우는 함수. 평론가 먼저, 배우 나중."""

    s, a, r, s_next, done = buffer.sample(batch_size)   # 128개 꺼내기

    # ── ① 평론가 배우기 ──
    with torch.no_grad():                      # 정답 만들기 — 미분 금지 구역
        # 다음 상황에서 '과녁 배우'가 할 행동을 구하고,
        # 그 행동의 점수를 '과녁 평론가'에게 물어본다.
        target_q = critic_t(s_next, actor_t(s_next))

        # 정답 = 지금 받은 점수 + 감마 x 다음 상황의 값
        # (1 - done) : 판이 끝났으면 뒤는 없다
        y = r + gamma * target_q * (1 - done)

    # 평론가의 예상이 정답에 가까워지게 한다
    critic_loss = nn.functional.mse_loss(critic(s, a), y)
    critic_opt.zero_grad()                     # 지난 기울기 지우기
    critic_loss.backward()                     # 어디를 고칠지 계산
    critic_opt.step()                          # 한 걸음 이동

    # ── ② 배우 배우기 ──
    # 배우의 목표는 "평론가에게 좋은 점수를 받는 행동을 내놓는 것" 입니다.
    #   critic(s, actor(s)) = 배우가 낸 행동을 평론가가 채점한 점수
    #   그 점수를 '키우고' 싶으니 마이너스를 붙여 '줄이는' 문제로 바꿉니다.
    actor_loss = -critic(s, actor(s)).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    # 여기서 평론가도 같이 바뀌지 않나요?
    #   critic_opt 는 평론가 것만 고치고 actor_opt 는 배우 것만 고칩니다.
    #   옵티마이저가 나뉘어 있어서 서로 건드리지 않습니다.

    # ── ③ 과녁을 아주 조금 따라오게 한다 ──
    soft_update(actor_t, actor)                # 0.5% 씩만 섞는다
    soft_update(critic_t, critic)


returns = []                                   # 판마다 점수 기록 (= 학습곡선)

for episode in range(200):                     # 200판을 한다
    s, _ = env.reset()                         # 새 판 시작
    total, done = 0.0, False

    while not done:
        # ── 행동 고르기 ──
        with torch.no_grad():                  # 행동만 고를 땐 미분 준비 불필요
            a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()

        # 배우는 같은 상황에서 항상 같은 답을 냅니다 (결정적).
        # 그대로 두면 새로운 것을 전혀 안 해봅니다 = 탐험이 없습니다.
        # 그래서 답에 살짝 흔들림을 더합니다. "32도" 대신 "32.4도" 처럼.
        #   어제 eps-greedy 가 "가끔 아예 딴 행동"이었다면,
        #   오늘은 "늘 조금씩 다르게" 입니다. 연속 행동이라 이 방식이 맞습니다.
        a += np.random.normal(0, noise_std * max_action, action_dim)

        # 흔들림을 더하다 범위를 벗어날 수 있으니 잘라 준다
        a = a.clip(-max_action, max_action)

        s_next, r, term, trunc, _ = env.step(a)    # 실제로 해본다
        done = term or trunc

        buffer.push(s, a, r, s_next, float(term))  # 일기장에 적는다
        s, total = s_next, total + r               # 다음 상황으로, 점수 누적

        if len(buffer) >= 1000:                # 1000줄 이상 쌓인 뒤부터 배운다
            train_step()

    returns.append(total)                      # 이번 판 점수 기록

    if episode % 10 == 0:                      # 10판마다 출력
        print(f"ep {episode:3d}  최근 10ep 평균 리턴 {np.mean(returns[-10:]):7.1f}")

# -1500 근처에서 시작해 -200 근처까지 올라오면 성공입니다.
# 학습에 6분쯤 걸립니다 — 돌려 놓고 설명을 들으시면 됩니다.
