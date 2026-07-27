# ============================================================
# [3일차 1교시 실습] 과녁을 통째로 갈까, 조금씩 섞을까
# ------------------------------------------------------------
# 어제(2일차) : 20판마다 과녁을 통째로 갈아 끼웠습니다 — 계단처럼 툭 바뀜
# 오늘(3일차) : 매번 0.5%씩만 섞습니다 — 미끄러지듯 천천히
#
# 왜 바꾸는지를 숫자와 그림으로 봅니다.
# 강화학습 코드가 아니라 아주 단순한 "따라가기" 문제로만 봅니다.
#
# 실행 시간: 3초
# ============================================================
import numpy as np

STEPS = 300                                # 300걸음 동안 따라가 본다
TAU = 0.005                                # 소프트 업데이트 비율 (0.5%)
HARD_EVERY = 50                            # 하드 업데이트 주기 (50걸음마다)


# ── 따라갈 대상(본체)이 계속 움직인다고 하자 ──────────────
# 학습 중인 신경망은 값이 계속 바뀝니다. 그걸 흉내 낸 것입니다.
rng = np.random.default_rng(0)
online = np.zeros(STEPS)                   # 본체의 값
v = 0.0
for t in range(STEPS):
    v += 0.02 + rng.normal(0, 0.05)        # 조금씩 오르면서 흔들린다
    online[t] = v

# ── 두 방식으로 과녁을 따라가게 한다 ──────────────────────
soft = np.zeros(STEPS)                     # 소프트: 매번 조금씩
hard = np.zeros(STEPS)                     # 하드: 가끔 통째로

s_val, h_val = 0.0, 0.0
for t in range(STEPS):
    # 소프트 업데이트: 새 과녁 = 0.5% 는 본체 + 99.5% 는 원래 과녁
    s_val = TAU * online[t] + (1 - TAU) * s_val
    soft[t] = s_val

    # 하드 업데이트: 주기가 되면 통째로 복사, 아니면 그대로
    if t % HARD_EVERY == 0:
        h_val = online[t]
    hard[t] = h_val


# ── 숫자로 보기 ───────────────────────────────────────────
print('=' * 62)
print('과녁이 본체를 얼마나 잘 따라가나')
print('=' * 62)
print('   걸음    본체      소프트(0.5%씩)   하드(50걸음마다)')
print('  ' + '-' * 56)
for t in range(0, STEPS, 40):
    print(f'  {t:5d}  {online[t]:8.3f}   {soft[t]:12.3f}   {hard[t]:14.3f}')

# 얼마나 갑자기 튀는지 재 본다 (걸음 사이의 변화량)
soft_jump = np.abs(np.diff(soft))
hard_jump = np.abs(np.diff(hard))

print(f'''
  === 과녁이 한 걸음에 얼마나 움직였나 ===
  소프트   평균 {soft_jump.mean():.4f}   가장 큰 변화 {soft_jump.max():.4f}
  하드     평균 {hard_jump.mean():.4f}   가장 큰 변화 {hard_jump.max():.4f}

  → 하드는 평소엔 전혀 안 움직이다가 갱신 순간에 확 튑니다.
    가장 큰 변화가 소프트의 {hard_jump.max() / soft_jump.max():.0f}배쯤 됩니다.
  → 연속 행동은 값이 예민해서, 이렇게 튀면 학습이 그때마다 흔들립니다.
    그래서 3일차부터는 소프트 업데이트를 씁니다.''')

print(f'''
  === 그런데 따라가는 속도는? ===
  마지막 시점 기준 본체와의 거리
    소프트 {abs(online[-1] - soft[-1]):.3f}
    하드   {abs(online[-1] - hard[-1]):.3f}

  → 소프트는 부드럽지만 뒤처집니다. 공짜가 아닙니다.
    tau 를 키우면 빨리 따라오지만 다시 튀게 됩니다. 그 사이를 고르는 것입니다.''')


# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(online, color='k', lw=1, alpha=.5, label='online (moving)')
    ax[0].plot(soft, color='tab:blue', lw=2, label=f'soft (tau={TAU})')
    ax[0].plot(hard, color='tab:red', lw=2, label=f'hard (every {HARD_EVERY})')
    ax[0].set_xlabel('step'); ax[0].set_ylabel('value')
    ax[0].set_title('How the target follows'); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(soft_jump, color='tab:blue', lw=1.5, label='soft')
    ax[1].plot(hard_jump, color='tab:red', lw=1.5, label='hard')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('step'); ax[1].set_ylabel('jump size (log)')
    ax[1].set_title('How much it jumps each step'); ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout(); plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) TAU = 0.05 로 키우면? → 빨리 따라오지만 흔들림도 커집니다
#   2) TAU = 0.001 로 줄이면? → 아주 부드럽지만 많이 뒤처집니다
#   3) HARD_EVERY = 5 로 줄이면? → 소프트와 비슷해집니다
#      (자주 갈수록 "통째로 갈기"의 단점이 줄어듭니다)
#   4) 어제 2일차 코드의 TARGET_SYNC 를 떠올려 보세요. 같은 이야기입니다.
# ============================================================
