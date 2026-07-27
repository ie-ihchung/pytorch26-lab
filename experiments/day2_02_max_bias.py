# ============================================================
# [2일차 2교시 실습] 왜 Q값이 부풀려지는가 — 최대화 편향을 숫자로
# ------------------------------------------------------------
# 신경망도 강화학습도 쓰지 않습니다. 주사위만 던집니다.
# 그런데 Double DQN 이 왜 필요한지가 이 한 장으로 끝납니다.
#
# 상황 설정:
#   행동이 K개 있고, 사실은 전부 똑같이 별로입니다 (진짜 가치 = 0).
#   우리는 각 행동을 n번씩 해보고 평균을 내서 Q를 추정합니다.
#   추정에는 당연히 오차가 섞입니다. 어떤 건 운 좋게 +0.3, 어떤 건 -0.2.
#
# 여기서 max 를 취하면 무슨 일이 벌어질까요?
#   E[max(추정)] > max(E[추정]) = 0
#   → "가장 운 좋았던 놈"을 고르게 되므로 항상 0보다 큽니다.
#   이것이 최대화 편향(maximization bias)입니다. 실수가 아니라 구조입니다.
#
# 실행 시간: 2초
# ============================================================
import numpy as np

rng = np.random.default_rng(0)      # 결과 고정
TRIALS = 20000                      # 같은 실험을 2만 번 반복해 평균을 본다


def one_round(K, n):
    """
    K개 행동, 각각 n번 시도. 진짜 가치는 전부 0.
    반환: (그냥 max 로 추정한 값, Double 방식으로 추정한 값)
    """
    # 한 번 해볼 때마다 표준정규 잡음. 진짜 가치 0에 잡음만 얹힌 셈.
    samples = rng.standard_normal((K, n))         # (행동 K개, 각 n번)

    # ── ① 그냥 하는 방법 (Q-Learning 의 max) ──────────────
    q = samples.mean(axis=1)                      # 각 행동의 평균 = 추정 Q
    single = q.max()                              # 그 중 가장 큰 값을 그대로 믿는다

    # ── ② Double 방식 ────────────────────────────────────
    # 시도를 절반씩 두 묶음으로 나눈다.
    half = n // 2
    qA = samples[:, :half].mean(axis=1)           # 앞 절반으로 만든 추정
    qB = samples[:, half:].mean(axis=1)           # 뒤 절반으로 만든 추정
    best = qA.argmax()                            # 고르기는 A가 한다
    double = qB[best]                             # 점수는 B가 매긴다 (다른 눈으로)

    return single, double


print('진짜 가치는 전부 0 입니다. 추정치가 0에서 얼마나 벗어나는지 봅니다.\n')
print('  행동수 K | 시도 n |   그냥 max    |  Double 방식')
print('  ---------+--------+---------------+--------------')

rows = []
for K in [2, 5, 10, 50]:
    s = np.mean([one_round(K, 20)[0] for _ in range(TRIALS // 10)])
    d = np.mean([one_round(K, 20)[1] for _ in range(TRIALS // 10)])
    rows.append((K, s, d))
    print(f'  {K:8d} | {20:6d} | {s:+13.4f} | {d:+12.4f}')

print('\n  → 그냥 max 는 항상 0보다 큽니다. 행동이 많을수록 더 부풀려집니다.')
print('  → Double 방식은 0 근처에 머뭅니다. 고르는 눈과 채점하는 눈을 나눴기 때문입니다.')

# ── 시도 횟수를 늘리면 줄어드는가 ─────────────────────────
print('\n  같은 K=10 에서 시도 횟수 n 만 늘려 보면:')
print('  시도 n |   그냥 max    |  Double 방식')
print('  -------+---------------+--------------')
ns, ss, ds = [], [], []
for n in [4, 10, 20, 50, 100, 200]:
    s = np.mean([one_round(10, n)[0] for _ in range(2000)])
    d = np.mean([one_round(10, n)[1] for _ in range(2000)])
    ns.append(n); ss.append(s); ds.append(d)
    print(f'  {n:6d} | {s:+13.4f} | {d:+12.4f}')

print('\n  → 많이 해볼수록 줄어들긴 합니다. 하지만 0으로 가는 속도가 느립니다.')
print('  → 강화학습은 "충분히 많이" 해볼 수 없는 상황이 대부분이라 이 편향이 남습니다.')

# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    Ks = [r[0] for r in rows]
    ax[0].bar([i - 0.2 for i in range(len(Ks))], [r[1] for r in rows], width=0.4, label='plain max')
    ax[0].bar([i + 0.2 for i in range(len(Ks))], [r[2] for r in rows], width=0.4, label='double')
    ax[0].axhline(0, color='k', lw=1)                       # 진짜 값 = 0
    ax[0].set_xticks(range(len(Ks))); ax[0].set_xticklabels([f'K={k}' for k in Ks])
    ax[0].set_ylabel('estimated value (true = 0)')
    ax[0].set_title('More actions -> more overestimation'); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(ns, ss, 'o-', label='plain max')
    ax[1].plot(ns, ds, 's-', label='double')
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xscale('log'); ax[1].set_xlabel('samples per action (n)')
    ax[1].set_ylabel('estimated value (true = 0)')
    ax[1].set_title('Bias shrinks slowly'); ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 수식으로 한 줄
#   E[ max_a Q̂(a) ]  ≥  max_a E[ Q̂(a) ]  =  max_a Q(a)
#   (젠센 부등식 — max 는 볼록함수라 기댓값과 순서를 바꾸면 커집니다)
#   쉽게: "여러 번 재고 그중 최고를 고르면, 그 최고에는 운도 섞여 있다"
#
# Double DQN 이 하는 일
#   고르기: argmax 는 온라인 네트워크가
#   채점  : 그 행동의 값은 타깃 네트워크가
#   → 운 좋게 튄 값이 선택과 평가에 동시에 반영되지 않습니다.
#
# 바꿔 보기
#   1) 진짜 가치를 전부 0이 아니라 [0,0,0,...,+0.5] 로 두면?
#      → 편향 때문에 진짜 좋은 행동이 묻히는 상황을 만들 수 있습니다
#   2) 잡음 크기를 rng.standard_normal * 3 으로 키우면? → 편향도 3배
# ============================================================
