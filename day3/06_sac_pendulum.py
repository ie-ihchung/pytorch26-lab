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

import gymnasium as gym                        # 게임(환경) 만드는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구
import copy                                    # 통째로 복사하는 도구

env = gym.make("Pendulum-v1")                  # 막대 세우기 (연속 행동)
state_dim, action_dim = 3, 1                   # 상황 3개, 행동 1개
max_action = float(env.action_space.high[0])   # 힘의 최대 크기 = 2.0

actor = GaussianActor(state_dim, action_dim, max_action)   # 4교시에서 만든 확률적 배우

q1, q2 = Critic(state_dim, action_dim), Critic(state_dim, action_dim)
# ★ 평론가를 두 명 둡니다 (트윈 Q) ★
#   왜 두 명인가요?
#     한 명이면 운 좋게 높게 매긴 점수를 그대로 믿게 됩니다 (2일차 최대화 편향).
#     두 명에게 물어보고 '더 낮게 말한 쪽'을 택하면 부풀려지는 걸 막습니다.
#     보수적으로 보는 것이 안전하다는 뜻입니다.

q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)   # 각각의 과녁(복사본)

actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)          # 배우용
q_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=3e-4)
# 평론가 둘을 한 옵티마이저로 함께 학습시킵니다 (손실도 더해서 한 번에)

# ── 온도 alpha 를 사람이 정하지 않고 스스로 조절하게 만든다 ──
log_alpha = torch.zeros(1, requires_grad=True)   # alpha 의 로그값을 학습 대상으로
                                                 # 로그로 두는 이유: alpha 는 양수여야 하므로
alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)

target_entropy = -action_dim                   # 목표 엔트로피 = -(행동 개수) = -1
# 이 값이 뭔가요?
#   "이 정도는 계속 헷갈려 하자" 는 기준선입니다.
#   실제 엔트로피가 이보다 낮아지면(너무 확신하면) alpha 를 키워 탐험을 늘리고,
#   높아지면(너무 헤매면) alpha 를 줄입니다. 자동 온도조절기입니다.

buffer = ReplayBuffer(100_000, action_dtype=torch.float32)   # 연속 행동이므로 float32
gamma, batch_size = 0.99, 256                  # 한 번에 256개씩 꺼내 배운다


def train_step():
    """한 번 배우는 함수. 평론가 -> 배우 -> 온도 순서입니다."""

    s, a, r, s_next, done = buffer.sample(batch_size)

    alpha = log_alpha.exp().detach()           # 로그를 되돌려 실제 alpha 값으로
                                               # detach: 여기서는 alpha 를 상수로 쓴다

    # ── ① 평론가 둘 배우기 ──
    with torch.no_grad():                      # 정답 만들기 — 미분 금지 구역
        a_next, logp_next = actor(s_next)      # 다음 상황에서 할 행동과 그 로그확률

        q_next = torch.min(q1_t(s_next, a_next), q2_t(s_next, a_next))
        # 두 과녁 평론가 중 더 낮게 본 쪽을 택한다 (보수적으로)

        y = r + gamma * (1 - done) * (q_next - alpha * logp_next)
        #                              ^^^^^^^^^^^^^^^^^^^^^^^^^ 이게 SAC 의 핵심
        # 원래는 q_next 만 썼는데, 여기에 "얼마나 헷갈려 했는지"를 더해 줍니다.
        #   logp 가 작다(=확률이 낮다=뜻밖의 행동을 했다) -> 빼는 값이 커짐 -> 보너스
        # 즉 "골고루 해보는 것 자체에 점수를 준다"는 뜻입니다.

    q_loss = nn.functional.mse_loss(q1(s, a), y) + nn.functional.mse_loss(q2(s, a), y)
    # 두 평론가의 손실을 더해 한 번에 학습시킨다
    q_opt.zero_grad(); q_loss.backward(); q_opt.step()

    # ── ② 배우 배우기 ──
    a_new, logp = actor(s)                     # 지금 정책으로 행동을 다시 뽑아 본다
                                               # (일기장에 적힌 옛날 행동이 아니라 새로!)
    q_new = torch.min(q1(s, a_new), q2(s, a_new))     # 두 평론가 중 낮은 쪽

    actor_loss = (alpha * logp - q_new).mean()
    # 배우의 목표 두 가지를 한 줄에 담았습니다.
    #   -q_new  : 점수를 높이고 싶다
    #   +alpha*logp : 너무 확신하지 말라 (logp 가 크면 = 확신하면 손실이 커짐)
    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

    # ── ③ 온도 alpha 스스로 조절하기 ──
    alpha_loss = -(log_alpha.exp() * (logp + target_entropy).detach()).mean()
    # 읽는 법:
    #   logp + target_entropy 가 양수 = 너무 확신하고 있다 -> alpha 를 키운다
    #   음수 = 충분히 헤매고 있다 -> alpha 를 줄인다
    # .detach() 로 배우 쪽에는 영향을 주지 않게 끊었습니다.
    alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()

    # ── ④ 과녁을 조금씩 따라오게 ──
    soft_update(q1_t, q1); soft_update(q2_t, q2)
    # 배우에는 과녁이 없습니다. SAC 는 배우 과녁이 필요 없기 때문입니다.


returns = []                                   # 판별 점수 (= 학습곡선)

for episode in range(150):                     # 150판
    s, _ = env.reset()
    total, done = 0.0, False

    while not done:
        with torch.no_grad():                  # 행동만 고를 땐 미분 불필요
            a, _ = actor(torch.as_tensor(s, dtype=torch.float32))
            # 로그확률은 지금 필요 없으니 _ 로 버립니다.
            # DDPG 와 달리 잡음을 따로 안 더합니다 — 정책 자체가 확률적이니까요.

        s_next, r, term, trunc, _ = env.step(a.numpy())    # 실제로 해본다
        done = term or trunc

        buffer.push(s, a.numpy(), r, s_next, float(term))  # 일기장에 적기
        s, total = s_next, total + r

        if len(buffer) >= 1000:                # 1000줄 쌓인 뒤부터 배운다
            train_step()

    returns.append(total)

    if episode % 10 == 0:
        print(f"ep {episode:3d}  평균 {np.mean(returns[-10:]):7.1f}  alpha {log_alpha.exp().item():.3f}")
        # alpha 가 어떻게 변하는지 함께 보세요.
        # 보통 처음엔 커졌다가(많이 탐험) 나중엔 작아집니다(확신이 생김).

# 학습에 9분쯤 걸립니다. -1200 에서 시작해 -200 근처까지 오르면 잘 된 것입니다.
