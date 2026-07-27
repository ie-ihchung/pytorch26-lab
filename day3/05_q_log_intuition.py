# ==========================================================
# 3일차 5교시 — TAC 소개
# 2026-07-29 (수) 14:30 ~ 15:30 · Advanced Actor-Critic Methods
# 원본 파일명: q_log_intuition.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s5
# ==========================================================
# [학습목표]
#  - 샤논 엔트로피의 한계와 Tsallis 엔트로피의 일반화를 이해한다
#  - TAC의 q 파라미터가 탐험 스타일에 주는 영향을 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import torch

def q_log(x, q):
    """Tsallis q-logarithm: q→1이면 자연로그로 수렴"""
    if abs(q - 1.0) < 1e-6:
        return torch.log(x)
    return (x.pow(q - 1) - 1) / (q - 1)

# 같은 Q값에 대해 q에 따라 정책 분포가 어떻게 달라지는가 (수치 근사)
q_values = torch.tensor([1.0, 0.9, 0.2, -1.0])
alpha = 0.5

for q in [1.0, 1.5, 2.0]:
    # 정책 최적화를 경사하강으로 근사: max E[Q] + alpha * S_q(pi)
    logits = torch.zeros(4, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(2000):
        pi = torch.softmax(logits, dim=0)
        entropy_q = -(pi * q_log(pi, q)).sum()
        loss = -((pi * q_values).sum() + alpha * entropy_q)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"q={q:.1f}  pi={pi.detach().numpy().round(3)}")

# q=1.0 → 모든 행동에 확률 배분 (SAC와 동일)
# q=2.0 → 나쁜 행동(Q=-1)의 확률이 사실상 0으로 — sparse한 탐험
