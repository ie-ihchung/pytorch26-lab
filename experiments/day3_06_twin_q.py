# ============================================================
# [3일차 6교시 실습] 평론가를 두 명 두면 뭐가 좋아지나 (트윈 Q)
# ------------------------------------------------------------
# 어제 2교시에서 "여럿 중 최고를 고르면 운도 함께 뽑힌다"를 봤습니다.
# 오늘 SAC 는 다른 방법으로 같은 문제를 막습니다.
#
#   평론가를 두 명 두고, 둘 중 '더 낮게 말한 쪽'을 택한다.
#
# 정말 효과가 있는지 주사위로 확인합니다. 신경망은 안 씁니다.
#
# 실행 시간: 3초
# ============================================================
import numpy as np

rng = np.random.default_rng(0)             # 결과 고정
TRIALS = 20000                             # 같은 실험을 2만 번 반복


def one_round(K, noise):
    """
    행동 K개. 진짜 가치는 전부 0.
    평론가들이 그 값을 추정하는데 잡음이 섞인다.
    반환: (평론가 1명일 때, 두 명 중 낮은 쪽을 쓸 때, 두 명 평균을 쓸 때)
    """
    # 평론가 A 와 B 가 각각 따로 추정한다 (잡음이 서로 다르다)
    qA = rng.normal(0, noise, K)
    qB = rng.normal(0, noise, K)

    # ── ① 평론가 한 명 (그냥 max) ──
    single = qA.max()

    # ── ② 두 명 중 낮은 쪽을 쓰고, 그중 max ── (SAC 방식)
    twin = np.minimum(qA, qB).max()

    # ── ③ 두 명의 평균을 쓰고, 그중 max ── (비교용)
    avg = ((qA + qB) / 2).max()

    return single, twin, avg


print('=' * 64)
print('진짜 가치는 전부 0 입니다. 추정치가 0에서 얼마나 벗어나는지 봅니다.')
print('=' * 64)
print('  행동수 K |  평론가 1명  |  두 명 중 낮은 쪽  |  두 명 평균')
print('  ---------+--------------+--------------------+------------')

rows = []
for K in [2, 5, 10, 50]:
    s = np.mean([one_round(K, 1.0)[0] for _ in range(TRIALS // 10)])
    t = np.mean([one_round(K, 1.0)[1] for _ in range(TRIALS // 10)])
    a = np.mean([one_round(K, 1.0)[2] for _ in range(TRIALS // 10)])
    rows.append((K, s, t, a))
    print(f'  {K:8d} | {s:+12.4f} | {t:+18.4f} | {a:+10.4f}')

print('''
  → 평론가 1명은 항상 0보다 큽니다 (어제 본 최대화 편향).
  → 두 명 중 낮은 쪽을 쓰면 그 부풀림이 절반 이하로 줄어듭니다.
    행동이 2개일 때는 아예 0보다 작아집니다(-0.08). 보수적으로 보는 것입니다.
    다만 행동이 많아지면 여전히 0보다 큽니다 — ==완전히 없애지는 못합니다.==
  → 평균을 쓰면 줄어드는 폭이 훨씬 작습니다. 그래서 평균이 아니라 최솟값을 씁니다.

  왜 평균이 아니라 '낮은 쪽'을 쓸까요?
    강화학습에서는 과대평가가 과소평가보다 훨씬 위험합니다.
    과대평가하면 "저기 좋아 보인다"며 나쁜 길로 계속 갑니다.
    과소평가는 조금 소극적일 뿐, 스스로 고쳐집니다.''')


print()
print('=' * 64)
print('잡음이 클수록 차이가 커지는지 확인')
print('=' * 64)
print('  잡음 크기 |  평론가 1명  |  두 명 중 낮은 쪽')
print('  ----------+--------------+------------------')

ns, ss, ts = [], [], []
for noise in [0.2, 0.5, 1.0, 2.0, 3.0]:
    s = np.mean([one_round(10, noise)[0] for _ in range(2000)])
    t = np.mean([one_round(10, noise)[1] for _ in range(2000)])
    ns.append(noise); ss.append(s); ts.append(t)
    print(f'  {noise:9.1f} | {s:+12.4f} | {t:+17.4f}')

print('''
  → 잡음이 클수록 1명일 때의 부풀림이 커집니다.
  → 학습 초반에는 평론가가 엉터리라 잡음이 큽니다.
    바로 그때 트윈 Q 가 가장 크게 도움이 됩니다.''')


# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    Ks = [r[0] for r in rows]
    idx = range(len(Ks))
    ax[0].bar([i - 0.25 for i in idx], [r[1] for r in rows], width=0.25, label='single critic')
    ax[0].bar([i for i in idx], [r[2] for r in rows], width=0.25, label='twin (min)')
    ax[0].bar([i + 0.25 for i in idx], [r[3] for r in rows], width=0.25, label='twin (mean)')
    ax[0].axhline(0, color='k', lw=1)
    ax[0].set_xticks(list(idx)); ax[0].set_xticklabels([f'K={k}' for k in Ks])
    ax[0].set_ylabel('estimate (true = 0)')
    ax[0].set_title('More actions -> more overestimation')
    ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(ns, ss, 'o-', label='single critic')
    ax[1].plot(ns, ts, 's-', label='twin (min)')
    ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel('noise level'); ax[1].set_ylabel('estimate (true = 0)')
    ax[1].set_title('Noisier estimates -> bigger bias')
    ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout(); plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 어제와 무엇이 다른가
#   어제 Double DQN : 고르는 사람과 채점하는 사람을 나눴다
#   오늘 트윈 Q     : 두 명에게 물어보고 낮은 쪽을 듣는다
#   목적은 같습니다 — 운 좋게 튄 값을 그대로 믿지 않기.
#
# 바꿔 보기
#   1) 평론가를 세 명으로 늘려 보세요 (np.minimum 을 세 번).
#      더 보수적이 되는데, 너무 과소평가하면 학습이 느려집니다.
#      강사 맥북 실측(K=10, 잡음 1.0): 1명 +1.54 / 2명 최솟값 +0.66 / 2명 평균 +1.10
#   2) 진짜 가치를 [0,0,...,+1.0] 으로 두면?
#      진짜 좋은 행동을 트윈 Q 가 놓치지는 않는지 확인해 보세요.
# ============================================================
