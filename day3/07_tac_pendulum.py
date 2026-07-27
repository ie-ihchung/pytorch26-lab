# ==========================================================
# 3일차 7교시 — TAC 구현
# 2026-07-29 (수) 16:30 ~ 17:30 · Advanced Actor-Critic Methods
# 원본 파일명: tac_pendulum.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/3#s7
# ==========================================================
# [학습목표]
#  - SAC 코드에서 엔트로피 항을 Tsallis q-log로 교체해 TAC를 완성한다
#  - q 값을 바꿔가며 탐험 스타일의 변화를 관찰한다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day3_전체.ipynb를 위에서부터 실행하세요.

import torch

# SAC → TAC: 엔트로피 항의 log를 q-log로 교체하는 것이 전부
ENTROPIC_INDEX = 2.0        # q=1.0이면 SAC와 동일

def q_log_prob(log_prob, q=ENTROPIC_INDEX):
    """log π → log_q π 변환 (log_prob은 SAC actor가 주는 값)"""
    if abs(q - 1.0) < 1e-6:
        return log_prob                       # q→1: SAC로 환원
    prob = log_prob.exp().clamp(min=1e-8)
    return (prob.pow(q - 1) - 1) / (q - 1)    # q-logarithm

# ── SAC train_step에서 딱 두 곳만 수정 ──

# ① soft TD 목표 (수정 전: q_next - alpha * logp_next)
#    y = r + gamma * (1 - done) * (q_next - alpha * q_log_prob(logp_next))

# ② Actor 손실 (수정 전: alpha * logp - q_new)
#    actor_loss = (alpha * q_log_prob(logp) - q_new).mean()

# 실험 과제:
#  1. ENTROPIC_INDEX = 1.0으로 SAC와 동일 결과가 나오는지 검증
#  2. q = 1.5, 2.0에서 학습 곡선 비교
#  3. Pendulum은 행동공간이 작아 차이가 작습니다 —
#     HalfCheetah 등 고차원 환경에서 q>1의 효과가 뚜렷해집니다

# 3일 전체 요약
# Day1: MDP·벨만 → DP(모델有) → MC/TD(모델無) → SARSA/Q-Learning
# Day2: Q테이블 → 신경망(DQN·DDQN) / 정책 직접 학습(PG → A2C)
# Day3: 연속 행동(DDPG) → 최대 엔트로피(SAC) → 일반화 엔트로피(TAC)
