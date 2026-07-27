"""추가 실습 03 — 1일차 4교시 확장 실험 3가지

수업 중 "여유가 있다면 해보세요" 로 안내한 세 가지를 한 파일에 담았습니다.
==실행만 하면 세 실험 결과가 순서대로 나옵니다.== 코드를 고칠 필요가 없습니다.

  실험 ① 감마(γ)를 낮추면 가치 지형이 어떻게 변하나
  실험 ② "의도대로 안 움직일 확률"을 넣으면 정책이 어떻게 바뀌나
  실험 ③ 정책반복과 가치반복, 계산량이 얼마나 다른가

■ 실행
    python 03_gridworld_experiments.py

CPU 로 몇 초면 끝납니다. 1일차 4교시 뒤에 바로 돌려 보시면 됩니다.

■ 4교시 코드와 다른 점
4교시는 전이가 결정적이라 P[s][a] 가 (다음상태, 보상) 하나였습니다.
실험 ② 를 하려면 여러 갈래가 필요해서, 여기서는 P[s][a] 를 목록으로 바꿨습니다.
  P[s][a] = [(확률, 다음상태, 보상), (확률, 다음상태, 보상), ...]
==미끄러질 확률을 0으로 두면 4교시와 똑같아집니다.== 그래서 하나로 합칠 수 있었습니다.
"""
import numpy as np

# ── 4x4 격자 세계 (1일차 2교시와 같습니다) ──────────────────
N = 4
n_states = N * N
n_actions = 4                      # 0=상, 1=하, 2=좌, 3=우
TERMINALS = [0, n_states - 1]      # 왼쪽 위, 오른쪽 아래가 목표
ARROWS = np.array(["↑", "↓", "←", "→"])


def move(s, a):
    """행동 a 를 했을 때 어느 칸으로 가는가. 벽이면 제자리."""
    r, c = divmod(s, N)
    if a == 0:
        r = max(r - 1, 0)
    elif a == 1:
        r = min(r + 1, N - 1)
    elif a == 2:
        c = max(c - 1, 0)
    elif a == 3:
        c = min(c + 1, N - 1)
    return r * N + c


def build_P(slip=0.0):
    """전이 표를 만듭니다.

    slip = 의도한 방향으로 안 갈 확률.
      slip=0.0  → 항상 의도대로 (4교시와 동일)
      slip=0.2  → 80% 는 의도대로, 나머지 20% 를 다른 세 방향에 나눠 줌
    """
    P = []
    for s in range(n_states):
        row = []
        for a in range(n_actions):
            if s in TERMINALS:
                row.append([(1.0, s, 0)])          # 끝난 칸은 그대로, 점수 없음
                continue
            outs = []
            for a2 in range(n_actions):
                prob = (1 - slip) if a2 == a else slip / 3
                outs.append((prob, move(s, a2), -1))
            row.append(outs)
        P.append(row)
    return P


def q_of(P, V, s, gamma):
    """상태 s 에서 각 행동의 값. 갈래가 여럿이면 확률로 가중평균합니다.

    4교시에서는 갈래가 하나라 그냥 r + gamma*V 였습니다.
    여기서는 진짜 기댓값 계산(Σ)이 됩니다 — 수식의 Σ 가 코드에 나타난 것입니다.
    """
    return [sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][a])
            for a in range(n_actions)]


def value_iteration(P, gamma=1.0, theta=1e-6, max_sweep=10_000):
    """가치 반복. 몇 번 훑었는지(sweeps)도 함께 돌려줍니다."""
    V = np.zeros(n_states)
    sweeps = 0
    while sweeps < max_sweep:
        sweeps += 1
        delta = 0.0
        for s in range(n_states):
            if s in TERMINALS:
                continue
            v = max(q_of(P, V, s, gamma))
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    policy = np.array([int(np.argmax(q_of(P, V, s, gamma))) for s in range(n_states)])
    return V, policy, sweeps


def policy_iteration(P, gamma=1.0, theta=1e-6, max_sweep=1_000):
    """정책 반복. 계획을 몇 번 고쳤는지(outer)와 내부 계산 총량도 셉니다."""
    policy = np.zeros(n_states, dtype=int)
    outer = 0
    total_sweeps = 0

    while True:
        outer += 1

        # 1) 평가 — 지금 계획대로 갔을 때 각 칸이 얼마나 좋은지
        V = np.zeros(n_states)
        for _ in range(max_sweep):
            total_sweeps += 1
            delta = 0.0
            for s in range(n_states):
                if s in TERMINALS:
                    continue
                v = sum(p * (r + gamma * V[s2]) for p, s2, r in P[s][policy[s]])
                delta = max(delta, abs(v - V[s]))
                V[s] = v
            if delta < theta:
                break

        # 2) 개선 — 더 좋은 행동이 있으면 계획을 바꿈
        stable = True
        for s in range(n_states):
            if s in TERMINALS:
                continue
            best = int(np.argmax(q_of(P, V, s, gamma)))
            if best != policy[s]:
                stable = False
                policy[s] = best

        if stable:
            return V, policy, outer, total_sweeps


def show_policy(policy, indent="      "):
    for row in ARROWS[policy].reshape(N, N):
        print(indent + " ".join(row))


def summary(V):
    """가치의 범위와 가운데 4칸 평균. 지형이 평평해지는지 보는 지표입니다."""
    center = V[[5, 6, 9, 10]]
    return f"범위 {V.min():7.2f} ~ {V.max():5.2f}   가운데4칸 평균 {center.mean():7.2f}"


# ══════════════════════════════════════════════════════════
def experiment1():
    print("=" * 62)
    print(" 실험 ① 감마(γ)를 낮추면 — 미래를 덜 챙기게 하면?")
    print("=" * 62)
    print("  γ가 작을수록 먼 미래의 -1점이 작게 반영됩니다.")
    print("  그래서 목표에서 멀든 가깝든 값이 비슷해집니다(지형이 평평해짐).\n")

    P = build_P(slip=0.0)
    for gamma in (1.0, 0.9, 0.5):
        V, policy, sweeps = value_iteration(P, gamma)
        print(f"  γ = {gamma}   {summary(V)}   (계산 {sweeps}회)")

    print("\n  ▸ 짚어볼 점")
    print("    숫자가 전체적으로 0에 가까워집니다. 가치 지형이 평평해진 것입니다.")
    print("    그럼 γ는 무조건 크면 좋을까요? 아닙니다.")
    print("    끝나지 않는 문제에서 γ=1이면 점수가 무한히 쌓여 계산이 안 됩니다.")
    print("    오늘 γ=1을 쓸 수 있는 건 이 문제가 언젠가 끝나기 때문입니다.")


def experiment2():
    print("\n" + "=" * 62)
    print(" 실험 ② 미끄러짐을 넣으면 — 의도대로 안 움직인다면?")
    print("=" * 62)
    print("  빙판길처럼 가끔 엉뚱한 방향으로 미끄러지게 만들어 봅니다.\n")

    results = {}
    for slip in (0.0, 0.2, 0.5):
        V, policy, sweeps = value_iteration(build_P(slip), gamma=1.0)
        results[slip] = (V, policy, sweeps)
        print(f"  미끄러질 확률 {slip:.0%}   {summary(V)}   (계산 {sweeps}회)")
        show_policy(policy)
        print()

    print("  ▸ 짚어볼 점")
    v0, v2 = results[0.0][0], results[0.2][0]
    s0, s2 = results[0.0][2], results[0.2][2]
    print(f"    ① 값이 나빠집니다 ({v0[[5,6,9,10]].mean():.2f} → {v2[[5,6,9,10]].mean():.2f}).")
    print("       미끄러지면 헛걸음을 하니 목표까지 더 걸립니다.")
    print(f"    ② 계산이 훨씬 오래 걸립니다 ({s0}회 → {s2}회).")
    print("       확실할 때는 금방 끝나는데, 불확실하면 여러 경우를 다 따져야 합니다.")
    print("    ③ 정책도 바뀝니다. 오른쪽 위 칸을 비교해 보세요.")
    print("       위험한 쪽을 피해 돌아가는 길을 택합니다.")
    print("       사람도 빙판길에서는 돌아갑니다.")
    print("\n    ※ 이것이 오후에 배울 SARSA가 절벽에서 떨어져 걷는 이유와 같은 발상입니다.")


def experiment3():
    print("\n" + "=" * 62)
    print(" 실험 ③ 정책반복 vs 가치반복 — 계산량이 얼마나 다른가")
    print("=" * 62)
    print("  '신중한 사람'과 '성급한 사람'을 숫자로 비교합니다.\n")

    P = build_P(slip=0.0)
    Vp, pol_p, outer, total_sweeps = policy_iteration(P)
    Vv, pol_v, vi_sweeps = value_iteration(P)

    print(f"  정책반복(PI)  계획 수정 {outer}회 / 내부 계산 총 {total_sweeps}번")
    print(f"  가치반복(VI)  계산 {vi_sweeps}번 (계획 수정이라는 개념이 없음)")
    print(f"\n  두 가치함수 일치: {np.allclose(Vp, Vv)}")

    print("\n  PI 정책:")
    show_policy(pol_p)
    print("  VI 정책:")
    show_policy(pol_v)

    print("\n  ▸ 짚어볼 점")
    print(f"    정책반복은 계획을 {outer}번만 고쳤습니다. 적습니다.")
    print(f"    그런데 한 번 고칠 때마다 평균 {total_sweeps // outer}번씩 계산했습니다.")
    print(f"    가치반복은 계획 수정 없이 {vi_sweeps}번 훑고 끝났습니다.")
    print("    '정확히 알고 움직인다' vs '대충 알고 자주 움직인다' 의 차이입니다.")
    print("\n    두 정책이 같게 나온 것도 확인해 주세요.")
    print("    방식은 달라도 같은 답에 도착한다는 것이 이 실험의 핵심입니다.")
    print("\n    ※ 내부 계산이 크게 나오는 이유: 첫 계획(전부 '위')은 맨 윗줄에서")
    print("      벽에 막혀 수렴하지 않아 상한까지 갑니다. 그래서 상한이 필요했습니다.")


if __name__ == "__main__":
    experiment1()
    experiment2()
    experiment3()
    print("\n" + "=" * 62)
    print(" 세 실험 완료")
    print("=" * 62)


# ══════════════════════════════════════════════════════════
# 더 해보기
# ══════════════════════════════════════════════════════════
# ① build_P(slip=0.8) 로 해보세요.
#    거의 마음대로 안 움직이는 상황입니다. 정책이 어떻게 변하는지 보세요.
#
# ② experiment1 의 감마 목록에 0.1 을 추가해 보세요.
#    거의 당장만 보는 것이라 지형이 아주 평평해집니다.
#
# ③ policy_iteration 의 max_sweep 을 1000 → 10 으로 줄여 보세요.
#    "대충 평가하고 자주 고치기"가 되어 가치반복에 가까워집니다.
#    계획 수정 횟수는 늘지만 총 계산량은 오히려 줄어드는 것을 볼 수 있습니다.
#
# ④ 격자 크기 N 을 4 → 6 으로 키워 보세요.
#    칸이 16개에서 36개로 늘어납니다. 계산이 얼마나 더 걸리는지 확인해 보세요.
#    상태가 조금만 늘어도 표 방식이 힘들어진다는 것을 체감할 수 있습니다.
#    (내일 신경망을 쓰는 이유가 바로 이것입니다.)
