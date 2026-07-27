# ============================================================
# [2일차 6교시 실습] 엔트로피 보너스 — 너무 일찍 확신하지 않게
# ------------------------------------------------------------
# A2C 손실은 이렇게 생겼습니다.
#   loss = 배우 손실 + 0.5 × 평론가 손실 − c × 엔트로피
#                                          ^^^^^^^^^^^^^
# 이 마지막 항이 오늘 실습 대상입니다.
#
# 엔트로피 = "얼마나 헷갈려 하는가"
#   두 행동 확률이 (0.5, 0.5) 면 엔트로피가 큽니다 (아직 모르겠다)
#   (0.99, 0.01) 이면 작습니다 (확신한다)
#
# 손실에서 엔트로피를 '빼면' → 엔트로피가 큰 쪽이 손실이 작아집니다
#   → 헷갈리는 상태를 조금 선호하게 됩니다 = 계속 탐험하게 됩니다
#
# c 를 0 / 0.01 / 0.1 세 가지로 바꿔 돌려 봅니다.
# 실행 시간: 2분 안팎 (CPU)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 0
UPDATES = 3000         # 몇 번 배울지 (7교시 수업 코드와 같은 값)
N_STEPS = 5            # 몇 걸음 모아서 한 번 배울지
GAMMA = 0.99
LR = 7e-4


class ActorCritic(nn.Module):
    """몸통은 함께 쓰고, 머리만 둘로 나눈다 (배우 머리 / 평론가 머리)."""
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_obs, 128), nn.ReLU())
        self.actor = nn.Linear(128, n_act)     # 무엇을 할지
        self.critic = nn.Linear(128, 1)        # 여기가 얼마나 좋은지

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h)


def run(entropy_coef):
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    s, _ = env.reset(seed=SEED)
    ep_ret, scores, ents = 0.0, [], []

    for _ in range(UPDATES):
        log_probs, values, rewards, dones, entropies = [], [], [], [], []

        # ── n걸음만 걸어 본다 ────────────────────────────
        for _ in range(N_STEPS):
            logits, v = model(torch.tensor(s, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())       # 얼마나 헷갈려 하는지
            values.append(v.squeeze())

            s2, r, term, trunc, _ = env.step(int(a))
            rewards.append(r)
            dones.append(float(term))
            ep_ret += r

            s = s2
            if term or trunc:                      # 한 판이 끝나면
                scores.append(ep_ret); ep_ret = 0.0
                s, _ = env.reset()

        # ── 마지막 지점의 값 (더 갈 수 있었다면 그 뒤는 이 값으로 대신) ──
        with torch.no_grad():                      # 목표는 상수여야 한다
            _, last_v = model(torch.tensor(s, dtype=torch.float32))
            R = last_v.squeeze()

        # ── 뒤에서부터 접어 오며 목표값 만들기 ────────────
        targets = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            targets.append(R)
        targets = torch.stack(list(reversed(targets))).detach()

        values = torch.stack(values)
        advantage = (targets - values).detach()    # 평론가로 신호가 역류하지 않게

        actor_loss = -(torch.stack(log_probs) * advantage).mean()
        critic_loss = nn.functional.mse_loss(values, targets)
        entropy = torch.stack(entropies).mean()
        ents.append(entropy.item())

        # ★ 이 줄이 오늘의 실습 ★
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

        opt.zero_grad(); loss.backward(); opt.step()

    env.close()
    return scores, ents


def smooth(x, k=20):
    return [np.mean(x[max(0, i - k):i + 1]) for i in range(len(x))]


results = {}
for c in [0.0, 0.01, 0.1]:
    print(f'엔트로피 계수 c = {c} 학습 중...')
    sc, en = run(c)
    results[c] = (sc, en)
    tail = sc[-30:] if len(sc) >= 30 else sc
    # 엔트로피는 한 번의 값이 너무 흔들린다 — 뒤쪽 200번 평균으로 본다
    print(f'   판 수 {len(sc):4d}   마지막 30판 평균 {np.mean(tail):6.1f}'
          f'   엔트로피(뒤 200회 평균) {np.mean(en[-200:]):.3f}')

# ── 결론은 미리 적어 두지 않고 이번 결과에서 뽑는다 ──────────
best = max(results, key=lambda c: np.mean(results[c][0][-30:]))
print(f'''
  === 이번 실행에서는 c = {best} 가 가장 높았습니다 ===

  엔트로피 값 읽는 법
    ln2 = 0.693 에 가까우면  "아직도 반반이다" (탐험 중)
    0 에 가까우면            "완전히 결정했다" (탐험 끝)

  ★ 여기서 배울 것은 "c 는 얼마가 정답"이 아닙니다 ★

  강사 맥북 실측(seed=0):  c=0 → 106.7점,  c=0.01 → 97.0점,  c=0.1 → 77.5점
  엔트로피(뒤 200회 평균)는 각각 0.605 / 0.583 / 0.580 으로 ==거의 같았습니다.==

  교과서 설명대로라면 c 를 키울수록 엔트로피가 높게 유지돼야 하는데
  이번 실험에서는 **차이가 거의 없었습니다.** 이유는 두 가지입니다.
    · CartPole 은 행동이 2개뿐이라 엔트로피가 커봐야 ln2 = 0.693 입니다
    · 15,000 스텝은 짧아서 정책이 아직 확신을 갖기 전입니다

  → 하이퍼파라미터는 **문제마다 다릅니다.** 남이 쓴 값을 그대로 믿지 말고
    이렇게 직접 돌려서 확인하는 습관이 실전에서 훨씬 중요합니다.
  → 한 번의 실행으로 "c=0 이 낫다"고 결론 내면 안 됩니다. 아래 2)번을 꼭 해보세요.
  → 엔트로피 보너스가 확실히 필요해지는 건 행동이 많고 함정이 있는
    문제입니다 (예: 미로에서 한 방향으로만 가려는 정책).''')

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for c, (sc, en) in results.items():
        ax[0].plot(smooth(sc), lw=2, label=f'entropy coef = {c}')
        ax[1].plot(smooth(en, 30), lw=2, label=f'entropy coef = {c}')
    ax[0].set_xlabel('episode'); ax[0].set_ylabel('return')
    ax[0].set_title('Learning curve'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].axhline(np.log(2), color='k', ls='--', lw=1)
    ax[1].set_xlabel('update'); ax[1].set_ylabel('policy entropy')
    ax[1].set_title('How undecided the policy stays (dashed = ln2)')
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# ============================================================
# 바꿔 보기
#   1) c = 0.5 로 크게 키우면? → 계속 딴짓을 해서 점수가 잘 안 오릅니다
#   2) SEED 를 1, 2, 3 으로 바꿔 세 번 더 돌려 보세요.
#      ★ 이게 이 실습에서 가장 중요합니다 ★
#      한 번의 결과로 "c=0 이 낫다"고 결론 내면 안 됩니다.
#      강화학습 논문들이 여러 seed 의 평균을 보고하는 이유입니다.
#   3) UPDATES 를 10000 으로 늘리면 순위가 바뀌는지 보세요
# ============================================================
