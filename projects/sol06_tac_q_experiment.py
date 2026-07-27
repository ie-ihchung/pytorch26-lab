# ==========================================================
# 미니 프로젝트 6 [심화] — TAC — 엔트로피 지수 q 실험
# 환경: Pendulum-v1 (심화: HalfCheetah-v5)
# 목표: Tsallis 엔트로피의 q 값이 탐험 스타일과 성능에 주는 영향을 검증한다
# 재사용: 3일차 7교시 TAC 코드 (SAC에서 q-log 두 줄 교체본)
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - 먼저 q=1.0으로 SAC와 결과가 일치하는지 회귀 테스트 (필수!)
#  - q ∈ {1.0, 1.5, 2.0}으로 학습곡선 비교
#  - 학습된 정책의 행동 분포 히스토그램을 q별로 비교 — q가 클수록 분포가 좁아지는가?
#  - Pendulum에서 차이가 작다면 왜 그런지(행동공간 크기) 생각해 보기
# [결과 인증]
#  q=1.0 ≒ SAC 검증 결과 + q별 학습곡선 비교 공유

# 전제: 6교시 SAC 학습 코드를 train_sac(entropic_index, seed) 함수로
#        감싸 두었음 (에피소드 리턴 리스트 반환). 수정 지점은 두 곳뿐.
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 그래프 한글 깨짐 방지 — 설치된 한글 폰트를 자동으로 골라 씁니다.
# Windows는 맑은 고딕, macOS는 AppleGothic이 기본 탑재입니다.
# Colab 등 리눅스는 !apt -qq install fonts-nanum 후 런타임을 다시 시작하세요.
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"):
    if _f in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

def q_log_prob(log_prob, q):
    """log pi → log_q pi (q=1이면 그대로 = SAC)"""
    if q is None or abs(q - 1.0) < 1e-6:
        return log_prob
    prob = log_prob.exp().clamp(min=1e-8)
    return (prob.pow(q - 1) - 1) / (q - 1)

# train_sac 내부에서 바꿀 두 곳:
#  ① soft TD 목표:  q_next - alpha * q_log_prob(logp_next, ENTROPIC_INDEX)
#  ② Actor 손실:    (alpha * q_log_prob(logp, ENTROPIC_INDEX) - q_new).mean()

# ── 실험 1: q=1.0 회귀 테스트 (SAC와 동일해야 함) ──
r_sac = train_sac(entropic_index=None, seed=0)   # 원본 SAC
r_q1 = train_sac(entropic_index=1.0, seed=0)     # TAC(q=1)
print("회귀 테스트 |평균 차이| =",
      abs(np.mean(r_sac[-20:]) - np.mean(r_q1[-20:])))   # 0 근처면 통과

# ── 실험 2: q별 학습곡선 (시드 3개 평균) ──
for q in [1.0, 1.5, 2.0]:
    runs = [train_sac(entropic_index=q, seed=sd) for sd in range(3)]
    plt.plot(np.mean(runs, axis=0), label=f"q={q}")
plt.xlabel("episode"); plt.ylabel("return"); plt.legend()
plt.title("TAC: entropic index q 비교")
plt.show()

# ── 실험 3: 행동 분포 히스토그램 (학습 완료된 actor 사용) ──
def action_hist(actor, label, n=3000):
    s = torch.randn(n, 3)                 # 임의 상태 배치
    with torch.no_grad():
        a, _ = actor(s)
    plt.hist(a.numpy().ravel(), bins=60, alpha=0.5, label=label, density=True)

# q별로 학습한 actor를 넣어 겹쳐 그리기 → q가 클수록 분포가 좁아지는지 확인
# action_hist(actor_q10, "q=1.0"); action_hist(actor_q20, "q=2.0")
# plt.legend(); plt.show()
