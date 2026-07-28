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

import gymnasium as gym                        # 게임(환경) 만드는 도구
import torch                                   # 파이토치
import torch.nn as nn                          # 신경망 부품 상자
import numpy as np                             # 숫자 계산 도구

env = gym.make("CartPole-v1")                  # 막대 세우기 게임
model = ActorCritic(4, 2)                      # 6교시에서 만든 배우+평론가 신경망
optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
# lr 이 DQN(1e-3)보다 조금 작습니다.
#   배우와 평론가가 옵티마이저 하나를 함께 쓰기 때문에 더 민감합니다.

gamma, n_steps = 0.99, 5
# gamma   = 미래를 얼마나 챙길지
# n_steps = 몇 걸음 걸어 보고 한 번 배울지
#   REINFORCE 는 한 판(수백 걸음)을 다 가야 배웠습니다 -> 느리고 들쭉날쭉
#   여기서는 5걸음마다 배웁니다 -> 자주, 안정적으로

s, _ = env.reset()                             # 첫 상황
ep_return, returns = 0, []
# ep_return = 지금 진행 중인 판의 누적 점수
# returns   = 끝난 판들의 점수 목록 (= 학습곡선)

for update in range(3000):                     # 3000번 배운다 (판 수가 아니라 '배운 횟수')

    # ── 1) n걸음만 걸어 보며 재료를 모은다 ──
    log_probs, values, rewards, entropies, dones = [], [], [], [], []

    for _ in range(n_steps):                   # 5걸음
        logits, v = model(torch.as_tensor(s, dtype=torch.float32))
        # logits = 배우가 낸 행동별 점수,  v = 평론가가 매긴 이 상황의 값

        dist = torch.distributions.Categorical(logits=logits)   # 점수 -> 확률분포
        a = dist.sample()                      # 확률대로 행동을 뽑는다

        s_next, r, term, trunc, _ = env.step(a.item())   # 실제로 해본다
        done = term or trunc                   # 쓰러졌거나 시간이 다 됐으면 끝

        log_probs.append(dist.log_prob(a))     # 그 행동의 로그확률 (나중에 미분할 값)
        values.append(v)                       # 평론가의 예상값
        rewards.append(r)                      # 받은 점수
        dones.append(done)                     # 여기서 판이 끝났나
        entropies.append(dist.entropy())       # 얼마나 헷갈려 했나

        ep_return += r                         # 이번 판 점수 누적
        s = s_next                             # 다음 상황으로 이동

        if done:                               # 판이 끝났으면
            returns.append(ep_return)          # 점수를 기록하고
            ep_return = 0                      # 초기화한 뒤
            s, _ = env.reset()                 # 새 판을 시작한다
            # 5걸음을 다 못 채워도 괜찮습니다. 이어서 새 판을 걷습니다.

    # ── 2) 목표값(정답)을 만든다 ──
    # 5걸음까지만 봤으니, 그 뒤는 평론가의 예상으로 대신합니다.
    with torch.no_grad():                      # 목표는 상수여야 한다 -> 미분 금지
        _, v_last = model(torch.as_tensor(s, dtype=torch.float32))

    R, td_targets = v_last, []
    for r, d in zip(reversed(rewards), reversed(dones)):   # 뒤에서부터 거꾸로
        R = r + gamma * R * (1 - d)            # 지금 점수 + 감마 x 뒤에서 온 값
                                               # (1-d) : 판이 끝난 자리에서는 뒤를 끊는다
        td_targets.insert(0, R)                # 앞에 끼워 원래 순서로

    td_targets = torch.stack(td_targets).detach()   # 낱개들을 하나로 쌓고 미분 끊기
    values = torch.stack(values)                    # 평론가 예상들도 하나로

    # ── 3) 어드밴티지 = 실제 - 예상 ──
    advantages = td_targets - values           # +면 예상보다 좋았다

    # ── 4) 손실 세 조각 ──
    actor_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
    # ★ advantages.detach() ★ 배우 쪽 신호가 평론가로 흘러가지 않게 끊는다

    critic_loss = advantages.pow(2).mean()     # 예상과 실제의 차이를 제곱해 평균
                                               # (= MSE 와 같습니다)
    entropy = torch.stack(entropies).mean()    # 평균적으로 얼마나 헷갈려 했나

    loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
    #                   ^^^ 평론가 손실은 커지기 쉬워 절반으로 누른다
    #                                      ^^^^ 엔트로피는 빼서 탐험을 조금 붙잡는다

    # ── 5) 배우기 ──
    optimizer.zero_grad()                      # 지난 기울기 지우기
    loss.backward()                            # 어디를 고칠지 계산

    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    # 기울기가 너무 크면 0.5 크기로 잘라 준다 (기울기 폭주 방지)
    #   가끔 한 번 크게 튀는 값 때문에 신경망이 통째로 망가지는 것을 막습니다.
    #   A2C 는 이게 없으면 학습 중에 갑자기 무너지는 일이 잦습니다.

    optimizer.step()                           # 한 걸음 이동

    if update % 200 == 0 and returns:          # 200번 배울 때마다 출력
        print(f"update {update:4d}  최근 20ep 평균 {np.mean(returns[-20:]):6.1f}")

# 3000번 배우면 대략 15,000걸음입니다.
#
# ★ 점수 편차가 아주 큽니다 ★
#   강사 맥북 실측: 같은 코드로 27점이 나온 적도, 107점이 나온 적도 있습니다.
#   A2C 는 시드(무작위 출발점)에 따라 결과가 크게 달라집니다.
#   한 번 돌려 보고 "안 되네" 하지 마시고 두세 번 돌려 보세요.
#   500점 만점을 보려면 range(3000) 을 30000 쯤으로 늘려야 합니다.
