# ============================================================
# [2일차 6교시 실습] detach() 를 빼면 무슨 일이 나는가
# ------------------------------------------------------------
# 같은 A2C 코드를 두 번 돌립니다. 다른 곳은 딱 한 군데입니다.
#
#     adv_for_actor = advantage.detach()   ← 있는 경우
#     adv_for_actor = advantage            ← 없는 경우
#
# ★ 이 실수의 무서운 점 ★
#   오류가 한 줄도 안 납니다. 그냥 조용히 나빠집니다.
#   초보자가 "손실이 줄어드니까 잘 되고 있네" 하고 넘어가기 딱 좋습니다.
#
# ------------------------------------------------------------
# detach() 가 하는 일 (한 문장)
#   "이 값은 참고만 하고, 여기까지 미분이 흘러오지는 마라"
#
# 왜 필요한가
#   평론가(Critic)의 일은 '값을 정확히 맞히는 것' 입니다.
#   그런데 detach 를 빼면, 배우(Actor) 손실을 줄이는 방향으로도
#   평론가가 끌려갑니다.
#
#   배우는 어드밴티지가 크면 좋아합니다.
#   어드밴티지 = 실제 - 평론가 예상  이므로,
#   평론가가 값을 '일부러 낮게' 부르면 어드밴티지가 커집니다.
#   → 정확도가 아니라 부정확한 쪽이 이득이 되어 버립니다.
#
#   그래서 평론가는 "정확히 맞히기" 대신
#   "배우가 좋아할 값 내놓기" 를 배우게 됩니다.
#
# 실행 시간: 약 35초 (3000번 x 2회)
# ============================================================
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 0
UPDATES = 3000          # 7교시 수업 코드와 같은 값
N_STEPS = 5             # 몇 걸음 모아서 한 번 배울지
GAMMA = 0.99            # 미래를 얼마나 챙길지
LR = 7e-4               # 학습률


class ActorCritic(nn.Module):
    """몸통은 함께 쓰고 머리만 둘 — 6교시에서 만든 그 구조입니다."""
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_obs, 128), nn.ReLU())
        self.actor = nn.Linear(128, n_act)      # 무엇을 할지
        self.critic = nn.Linear(128, 1)         # 여기가 얼마나 좋은지

    def forward(self, x):
        h = self.body(x)
        return self.actor(h), self.critic(h)


def run(use_detach):
    """use_detach=True 면 detach 를 쓰고, False 면 안 쓴다."""
    torch.manual_seed(SEED); np.random.seed(SEED)   # 두 실험의 출발점을 똑같이

    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    s, _ = env.reset(seed=SEED)
    ep_ret = 0.0
    scores = []                 # 판별 점수 (= 학습곡선)
    critic_losses = []          # 평론가 손실 기록 ← 이번 실습의 핵심 관찰 대상

    for _ in range(UPDATES):
        log_probs, values, rewards, dones, entropies = [], [], [], [], []

        # ── n걸음 걸어 보며 재료 모으기 ──────────────────
        for _ in range(N_STEPS):
            logits, v = model(torch.tensor(s, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()

            log_probs.append(dist.log_prob(a))
            entropies.append(dist.entropy())
            values.append(v.squeeze())

            s2, r, term, trunc, _ = env.step(int(a))
            rewards.append(r)
            dones.append(float(term))
            ep_ret += r
            s = s2

            if term or trunc:                   # 판이 끝나면 기록하고 새 판
                scores.append(ep_ret); ep_ret = 0.0
                s, _ = env.reset()

        # ── 목표값 만들기 (여기는 항상 no_grad — 정답은 과녁이니까) ──
        with torch.no_grad():
            _, last_v = model(torch.tensor(s, dtype=torch.float32))
            R = last_v.squeeze()

        targets = []
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + GAMMA * R * (1 - d)
            targets.append(R)
        targets = torch.stack(list(reversed(targets))).detach()

        values = torch.stack(values)
        advantage = targets - values            # 예상보다 얼마나 좋았나

        # ═══════════════════════════════════════════════════
        # ★★★ 이 두 줄이 이번 실습의 전부입니다 ★★★
        # ═══════════════════════════════════════════════════
        if use_detach:
            adv_for_actor = advantage.detach()  # 배우 쪽에서 평론가로 흐르는 길을 끊는다
        else:
            adv_for_actor = advantage           # 안 끊는다 (여기가 실수)
        # ═══════════════════════════════════════════════════

        actor_loss = -(torch.stack(log_probs) * adv_for_actor).mean()
        critic_loss = advantage.pow(2).mean()   # 평론가는 항상 자기 손실로 배운다
        entropy = torch.stack(entropies).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        critic_losses.append(critic_loss.item())

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

    env.close()
    return scores, critic_losses


def smooth(x, k=20):
    """들쭉날쭉한 값을 k개 이동평균으로 부드럽게"""
    return [np.mean(x[max(0, i - k):i + 1]) for i in range(len(x))]


results = {}
for label, ud in [('detach() 있음 (정상)', True), ('detach() 없음 (실수)', False)]:
    print(f'{label} 학습 중...')
    sc, cl = run(ud)
    results[label] = (sc, cl)
    tail = sc[-30:] if len(sc) >= 30 else sc
    print(f'   마지막 30판 평균 {np.mean(tail):6.1f}   최고 {max(sc):5.0f}   '
          f'평론가 손실 처음 {np.mean(cl[:100]):7.1f} → 끝 {np.mean(cl[-100:]):7.1f}')

print('''
============================================================
결과 읽는 법
============================================================

강사 맥북 실측 (seed=0)
                    점수      최고    평론가 손실(처음→끝)
  detach() 있음     17.2점    143       9.6 →  877
  detach() 없음      9.5점    113       9.6 → 2411

★ 두 가지를 보세요 ★

① 점수가 절반 아래로 떨어졌습니다.
   detach 하나 뺐을 뿐인데 학습이 눈에 띄게 나빠집니다.

② 평론가 손실이 세 배 가까이 커졌습니다.
   평론가가 값을 '더 못 맞히게' 된 것입니다.
   배우 쪽 신호가 흘러들어와 평론가가 다른 걸 배웠기 때문입니다.

③ 그런데 오류는 한 줄도 안 났습니다.
   그냥 조용히 나빠졌습니다. 이게 이 실수의 무서운 점입니다.

============================================================
여기서 꼭 짚고 갈 것 — 손실로 판단하면 안 됩니다
============================================================

두 경우 모두 평론가 손실이 9.6 에서 877 / 2411 로 '늘었습니다'.
줄어든 게 아닙니다. 왜 그럴까요?

  점수가 올라갔기 때문입니다.
  오래 버틸수록 목표값 자체가 커져서, 맞혀야 할 숫자가 커집니다.

→ 강화학습에서는 손실이 줄어드는 걸로 잘 되는지 판단하면 안 됩니다.
  점수(리턴)를 보셔야 합니다.

  지도학습만 해보신 분들이 가장 많이 헷갈리는 부분입니다.
  지도학습은 정답이 고정이라 손실이 줄면 잘 되는 것이 맞지만,
  강화학습은 정답(목표값) 자체가 학습과 함께 움직입니다.
''')

# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for label, (sc, cl) in results.items():
        color = 'tab:blue' if '있음' in label else 'tab:red'
        name = 'with detach()' if '있음' in label else 'without detach()'
        ax[0].plot(smooth(sc), lw=2, color=color, label=name)
        ax[1].plot(smooth(cl, 200), lw=2, color=color, label=name)

    ax[0].set_xlabel('episode'); ax[0].set_ylabel('return')
    ax[0].set_title('Score — higher is better'); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].set_yscale('log')
    ax[1].set_xlabel('update'); ax[1].set_ylabel('critic loss (log)')
    ax[1].set_title('Critic loss — lower means better value estimates')
    ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout(); plt.show()
except ImportError:
    print('  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) SEED 를 1, 2, 3 으로 바꿔 세 번 더 돌려 보세요.
#      ★ 이게 가장 중요합니다 ★
#      숫자는 매번 달라지지만 방향은 같습니다 —
#      detach 없는 쪽이 평론가 손실이 크고 점수가 낮습니다.
#      원리상 그럴 수밖에 없습니다.
#
#   2) 목표값 만드는 곳의 with torch.no_grad() 를 지워 보세요.
#      이번엔 '과녁이 도망가는' 경우입니다. 더 심하게 망가집니다.
#
#   3) critic_loss 만 따로 그려 보세요.
#      detach 없는 쪽이 처음부터 벌어지기 시작하는 것이 보입니다.
# ============================================================
