# ============================================================
# [2일차 5교시 실습] 베이스라인을 빼면 왜 학습이 안정되나
# ------------------------------------------------------------
# REINFORCE 를 두 번 돌립니다. 딱 한 줄만 다릅니다.
#
#   ① 그냥      : loss = -(log_prob * G).sum()
#   ② 베이스라인 : loss = -(log_prob * (G - G.mean()) / G.std()).sum()
#
# 왜 빼도 되나 (수학):
#   E[ ∇log π(a|s) · b ] = b · ∇( Σ_a π(a|s) ) = b · ∇1 = 0
#   상태에만 의존하는 값 b 는 기울기의 "평균"을 바꾸지 않습니다.
#   그런데 "흔들림(분산)"은 줄여 줍니다. 공짜로 얻는 안정성입니다.
#
# 쉽게:
#   시험 점수를 "80점"이라고만 하면 잘한 건지 모릅니다.
#   반 평균이 50점인지 90점인지 알아야 판단이 됩니다.
#   베이스라인 = 반 평균입니다.
#
# 실행 시간: 1~2분 (CPU)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 0
EPISODES = 400
GAMMA = 0.99
LR = 1e-3


class Policy(nn.Module):
    """상태를 받아 '각 행동을 할 점수(logit)'를 내놓는다."""
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        )

    def forward(self, x):
        return self.f(x)


def discounted_returns(rewards, gamma):
    """
    뒤에서부터 거꾸로 접어 오며 각 시점의 '앞으로 받을 총점' G 를 만든다.
      G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
    거꾸로 도는 이유: 앞에서부터 하면 매번 처음부터 다시 더해야 해서 느립니다.
    """
    out, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    return list(reversed(out))


def run(use_baseline):
    torch.manual_seed(SEED); np.random.seed(SEED)
    env = gym.make('CartPole-v1')
    pi = Policy(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(pi.parameters(), lr=LR)

    scores, grad_norms = [], []      # 점수 기록, 기울기 크기 기록(흔들림 측정)

    for ep in range(EPISODES):
        s, _ = env.reset(seed=SEED + ep)
        log_probs, rewards, done = [], [], False

        # ── 한 판을 끝까지 해본다 ────────────────────────
        while not done:
            logits = pi(torch.tensor(s))
            dist = torch.distributions.Categorical(logits=logits)   # 확률분포로 변환
            a = dist.sample()                                       # 확률대로 뽑는다
            log_probs.append(dist.log_prob(a))                      # 나중에 미분할 값
            s, r, term, trunc, _ = env.step(int(a))
            rewards.append(r)
            done = term or trunc

        # ── 판이 끝난 뒤 한 번에 배운다 ──────────────────
        G = torch.tensor(discounted_returns(rewards, GAMMA), dtype=torch.float32)

        if use_baseline:
            # ★ 이 한 줄이 전부입니다 ★
            G = (G - G.mean()) / (G.std() + 1e-8)   # 평균을 빼고 크기를 맞춘다

        loss = -(torch.stack(log_probs) * G).sum()
        opt.zero_grad()
        loss.backward()

        # 기울기가 얼마나 큰지 재둔다 (베이스라인 효과를 숫자로 보기 위함)
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in pi.parameters())).item()
        grad_norms.append(gn)

        opt.step()
        scores.append(sum(rewards))

        if (ep + 1) % 100 == 0:
            print(f'    에피소드 {ep+1:3d}  최근 100판 평균 {np.mean(scores[-100:]):6.1f}'
                  f'   기울기 크기 평균 {np.mean(grad_norms[-100:]):8.1f}')

    env.close()
    return scores, grad_norms


def smooth(x, k=20):
    return [np.mean(x[max(0, i - k):i + 1]) for i in range(len(x))]


print('① 베이스라인 없음 — 학습 중...')
s_no, g_no = run(use_baseline=False)
print('\n② 베이스라인 있음 — 학습 중...')
s_yes, g_yes = run(use_baseline=True)

print('\n  === 결과 ===')
print(f'  없음  마지막 100판 평균 {np.mean(s_no[-100:]):6.1f}   기울기 크기 {np.mean(g_no):9.1f}')
print(f'  있음  마지막 100판 평균 {np.mean(s_yes[-100:]):6.1f}   기울기 크기 {np.mean(g_yes):9.1f}')
print('\n  → 베이스라인을 쓰면 기울기 크기가 훨씬 작고 고릅니다.')
print('  → 기울기가 들쭉날쭉하면 학습률을 아무리 잘 잡아도 흔들립니다.')

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(s_no, alpha=.2, color='tab:red')
    ax[0].plot(s_yes, alpha=.2, color='tab:blue')
    ax[0].plot(smooth(s_no), color='tab:red', lw=2, label='no baseline')
    ax[0].plot(smooth(s_yes), color='tab:blue', lw=2, label='with baseline')
    ax[0].set_xlabel('episode'); ax[0].set_ylabel('return')
    ax[0].set_title('Learning curve'); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(smooth(g_no), color='tab:red', lw=2, label='no baseline')
    ax[1].plot(smooth(g_yes), color='tab:blue', lw=2, label='with baseline')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('episode'); ax[1].set_ylabel('gradient norm (log)')
    ax[1].set_title('How wildly the gradient swings'); ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout(); plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) G = G - G.mean() 만 하고 / G.std() 는 빼면? → 효과가 줄어듭니다
#   2) 베이스라인을 '고정값 100' 으로 두면? → 상태와 무관해도 평균은 안 바뀝니다
#      (그래서 편향은 안 생기지만, 분산 감소 효과는 작습니다)
#   3) 상태마다 다른 베이스라인 V(s) 를 신경망으로 배우면? → 그게 6교시 Actor-Critic 입니다
# ============================================================
