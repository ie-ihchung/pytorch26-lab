# ==========================================================
# 3일차 3교시 — Maximum Entropy RL 소개
# 2026-07-29 (수) 11:30 ~ 12:30 · Advanced Actor-Critic Methods
# 원본 파일명: entropy_intuition.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s3
# ==========================================================
# [학습목표]
#  - 최대 엔트로피 목적함수가 표준 RL 목적함수와 어떻게 다른지 이해한다
#  - 엔트로피 보너스가 탐험·강건성에 주는 효과를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import torch
import torch.nn.functional as F

# 엔트로피가 정책 분포에 주는 효과 직관 잡기
q_values = torch.tensor([1.0, 0.9, 0.2, -1.0])   # 행동 4개의 Q값

for alpha in [0.01, 0.5, 5.0]:
    # 최대 엔트로피 최적 정책: softmax(Q/alpha)
    pi = F.softmax(q_values / alpha, dim=0)
    entropy = -(pi * pi.log()).sum()
    print(f"alpha={alpha:4.2f}  pi={pi.numpy().round(3)}  H={entropy:.3f}")

# alpha=0.01 → 거의 greedy (첫 행동에 몰빵)
# alpha=0.5  → Q가 비슷한 행동 1,2를 골고루 선택  ← 여러 해를 포착
# alpha=5.0  → 거의 균등분포 (탐험 극대화)
