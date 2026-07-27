# ==========================================================
# 1일차 5교시 — Monte-Carlo 방법, Temporal Difference 방법 소개
# 2026-07-27 (월) 14:30 ~ 15:30 · Tabular-based Methods
# 원본 파일명: mc_vs_td_prediction.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/1#s5
# ==========================================================
# [학습목표]
#  - 모델 없이(model-free) 가치를 추정하는 두 접근을 이해한다
#  - MC의 낮은 편향·높은 분산, TD의 부트스트래핑·낮은 분산 특성을 비교한다
#  - TD(0) 업데이트 식을 쓸 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day1_전체.ipynb를 위에서부터 실행하세요.

import numpy as np                      # 숫자 계산을 도와주는 도구

# 같은 문제를 두 가지 방법으로 풀어 보고 결과를 비교합니다.
#   방법 1) 몬테카를로 — 한 판을 끝까지 하고 나서 배우기
#   방법 2) 시간차(TD)  — 한 걸음 옮길 때마다 바로 배우기

rng = np.random.default_rng(0)         # 무작위 뽑기 도구. 0은 '항상 같은 결과'를 위한 값
alpha = 0.05                           # 학습률 — 새로 안 것을 얼마나 믿을지 (0~1)
gamma = 1.0                            # 미래를 얼마나 챙길지 (1이면 먼 미래도 그대로)


def gen_episode():
    """한 판을 끝까지 해보고, 지나온 기록을 돌려줍니다."""
    s = rng.integers(1, n_states - 1)  # 출발 칸을 아무 데나 고름 (끝 칸 제외)
    episode = []                       # 지나온 기록을 담을 빈 목록

    while s not in TERMINALS:          # 끝 칸에 도착할 때까지 반복
        a = rng.integers(n_actions)    # 아무 방향이나 하나 고름 (무작위로 걷기)
        s_next, r = P[s][a]            # 그 방향으로 가면 어디로 가고 몇 점인지
        episode.append((s, r))         # "어느 칸에서 몇 점 받았다"를 기록
        s = s_next                     # 다음 칸으로 이동

    return episode                     # 한 판의 기록 전체를 돌려줌


# ── 방법 1) 몬테카를로 — 끝까지 하고 나서 배우기 ──────────
V_mc = np.zeros(n_states)              # 각 칸의 값. 처음엔 전부 0 (아무것도 모름)

for _ in range(5000):                  # 5000판 반복
    episode = gen_episode()            # 한 판을 끝까지 해봄
    G = 0.0                            # 이 지점부터 끝까지 받은 총 점수
    visited = set()                    # 이번 판에서 이미 들른 칸 목록

    # 뒤에서부터 되짚습니다. 끝에서부터 세야 "여기부터 끝까지"가 계산됩니다.
    for s, r in reversed(episode):
        G = r + gamma * G              # (지금 점수) + (여기 다음부터 끝까지)

        if s not in visited:           # 같은 칸을 여러 번 지났으면 첫 번째만 사용
            visited.add(s)             # 들렀다고 표시
            # 지금 알던 값(V_mc[s])을 실제 결과(G) 쪽으로 조금(alpha) 옮깁니다.
            V_mc[s] += alpha * (G - V_mc[s])


# ── 방법 2) 시간차(TD) — 한 걸음마다 바로 배우기 ──────────
V_td = np.zeros(n_states)              # 마찬가지로 전부 0에서 시작

for _ in range(5000):                  # 5000판 반복
    s = rng.integers(1, n_states - 1)  # 아무 칸에서 출발

    while s not in TERMINALS:          # 끝 칸에 닿을 때까지
        a = rng.integers(n_actions)    # 아무 방향이나 고름
        s_next, r = P[s][a]            # 가보니 어디로 갔고 몇 점인지

        # 여기가 몬테카를로와 다른 곳입니다.
        # 끝까지 안 가고, "지금 점수 + 다음 칸의 (아직 부정확한) 값"으로 대신합니다.
        td_error = r + gamma * V_td[s_next] - V_td[s]

        V_td[s] += alpha * td_error    # 그 차이만큼 조금 옮김
        s = s_next                     # 다음 칸으로
        s = s_next

print("MC 추정:"); print(np.round(V_mc.reshape(4, 4), 1))
print("TD 추정:"); print(np.round(V_td.reshape(4, 4), 1))
# 둘 다 DP 정답(-14, -20, -22...)에 근접하는지 확인하세요
