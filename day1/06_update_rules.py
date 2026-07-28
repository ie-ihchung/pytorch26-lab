# ==========================================================
# 1일차 6교시 — SARSA와 Q-Learning 소개
# 2026-07-27 (월) 15:30 ~ 16:30 · Tabular-based Methods
# 원본 파일명: update_rules.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s6
# ==========================================================
# [학습목표]
#  - TD 제어에서 SARSA와 Q-Learning의 업데이트 식을 구분한다
#  - On-policy와 Off-policy의 차이를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

# 두 알고리즘의 차이는 단 한 줄 — TD 목표(target)의 정의

# Q 는 '표'입니다. Q[칸][방향] = 그 칸에서 그 방향으로 가면 얼마나 좋은지.
# 아래 두 함수는 그 표의 숫자 하나를 고치는 방법입니다.
# 두 함수는 딱 한 줄만 다릅니다. 그 줄을 잘 보세요.


# ── 방법 1) SARSA — 내가 "실제로 할" 행동을 보고 고침 ──────
def sarsa_update(Q, s, a, r, s_next, a_next, alpha=0.1, gamma=0.99):
    # s      : 지금 있는 칸
    # a      : 지금 하려는 행동
    # r      : 그 행동으로 받은 점수
    # s_next : 그래서 가게 된 다음 칸
    # a_next : 다음 칸에서 "실제로 할" 행동   ← SARSA 는 이게 필요합니다
    # alpha  : 얼마나 믿을지 (0.1이면 10%만 반영)
    # gamma  : 미래를 얼마나 챙길지

    # 목표값 = 지금 받은 점수 + 다음 칸에서 실제로 할 행동의 값
    target = r + gamma * Q[s_next][a_next]

    # 지금 알던 값을 목표값 쪽으로 alpha 만큼 조금 옮깁니다.
    # (목표값 - 지금값) 이 '내 예상이 얼마나 빗나갔나' 입니다.
    Q[s][a] += alpha * (target - Q[s][a])


# ── 방법 2) Q-러닝 — "제일 좋은" 행동을 보고 고침 ─────────
def q_learning_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    # 여기는 a_next 가 없습니다. 다음에 뭘 할지 몰라도 되기 때문입니다.

    # 목표값 = 지금 받은 점수 + 다음 칸에서 "가장 좋은" 행동의 값
    # max 는 여러 값 중 제일 큰 것을 고르는 것입니다.
    # 실제로 그 행동을 할지는 상관하지 않습니다. "만약 최선을 다한다면" 을 가정합니다.
    target = r + gamma * max(Q[s_next])

    Q[s][a] += alpha * (target - Q[s][a])    # 고치는 방식은 위와 똑같습니다


# ── 행동 고르기 — 두 방법이 공통으로 씁니다 ───────────────
import random                                # 무작위 뽑기 도구


def epsilon_greedy(Q, s, n_actions, epsilon=0.1):
    # epsilon 은 '아무거나 해볼 확률' 입니다. 0.1이면 10번에 1번.
    # 왜 일부러 아무거나 할까요?
    #   처음 우연히 괜찮았던 길만 계속 가면 더 좋은 길을 영영 못 찾기 때문입니다.

    if random.random() < epsilon:            # 0~1 사이 아무 숫자를 뽑아서
        return random.randrange(n_actions)   # epsilon 보다 작으면 → 아무 방향이나

    # 그렇지 않으면 → 지금까지 알기로 가장 좋은 방향
    # key=... 는 "이 기준으로 가장 큰 것을 고르라" 는 뜻입니다.
    return max(range(n_actions), key=lambda a: Q[s][a])

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
print('두 갱신식이 어떻게 다른지 숫자로 확인합니다.')
print()

Q_demo = {0: [1.0, 5.0], 1: [3.0, 2.0]}      # 상태 2개, 행동 2개짜리 작은 표
r, gamma, alpha = 1.0, 0.9, 0.1
s, a, s2 = 0, 0, 1

a2_greedy = int(np.argmax(Q_demo[s2]))        # 가장 좋은 행동 (Q-러닝이 쓰는 것)
a2_actual = 1                                 # 실제로 하게 될 행동 (SARSA가 쓰는 것)

sarsa = Q_demo[s][a] + alpha * (r + gamma * Q_demo[s2][a2_actual] - Q_demo[s][a])
qlear = Q_demo[s][a] + alpha * (r + gamma * Q_demo[s2][a2_greedy] - Q_demo[s][a])

print(f'  다음 상태의 Q값 : {Q_demo[s2]}')
print(f'  SARSA  (실제 행동 {a2_actual}번 = {Q_demo[s2][a2_actual]}) -> 갱신값 {sarsa:.4f}')
print(f'  Q러닝  (최선 행동 {a2_greedy}번 = {Q_demo[s2][a2_greedy]}) -> 갱신값 {qlear:.4f}')
print()
print('  -> 같은 경험인데 값이 다릅니다. 딱 이 한 곳이 두 방법을 가릅니다.')
