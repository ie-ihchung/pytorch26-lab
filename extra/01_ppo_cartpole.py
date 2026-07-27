"""추가 실습 01 — PPO (Proximal Policy Optimization)

2일차 7교시 A2C 다음 단계입니다. 실무에서 가장 많이 쓰이는 방법입니다.

■ 왜 필요한가
A2C 는 한 번 모은 경험으로 한 번만 학습하고 버립니다. 아깝습니다.
그렇다고 같은 경험으로 여러 번 학습하면? 정책이 너무 많이 바뀌어 무너집니다.
"모은 경험이 지금 정책의 것"이라는 전제가 깨지기 때문입니다.

PPO 의 발상은 단순합니다.
  "여러 번 학습하되, 정책이 한 번에 너무 많이 바뀌지 않게 막자."

■ 어떻게 막나
'변화폭' 을 숫자로 잽니다. 같은 상황에서 같은 행동을 할 확률이
전보다 몇 배가 됐는지를 봅니다. 이것을 비율(ratio)이라 합니다.

  ratio = 지금 정책의 확률 / 경험을 모을 때 정책의 확률

이 값이 1 이면 안 바뀐 것, 1.5 면 1.5배 더 하게 된 것입니다.
PPO 는 이 비율을 0.8~1.2 같은 좁은 범위로 잘라(clip) 버립니다.
==범위를 벗어나면 더 이상 이득을 주지 않으니, 크게 바꿀 이유가 없어집니다.==

■ 실행
    python 01_ppo_cartpole.py
CPU 로 3~5분. CartPole 은 500점이 만점이고 475 넘으면 해결로 봅니다.
"""
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# ── 설정값 ──────────────────────────────────────────────────
GAMMA = 0.99          # 미래를 얼마나 챙길지
LAMBDA = 0.95         # GAE — 아래에서 설명
CLIP = 0.2            # 정책 변화 허용폭 (0.8배 ~ 1.2배)
EPOCHS = 10           # 모은 경험을 몇 번 재사용할지  ← A2C 는 1이었습니다
ROLLOUT = 2048        # 한 번에 모을 스텝 수
BATCH = 64
LR = 3e-4
TOTAL_STEPS = 100_000


def mlp(in_dim, out_dim):
    """작은 신경망 하나. 배우와 평론가가 같은 모양을 씁니다."""
    return nn.Sequential(
        nn.Linear(in_dim, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, out_dim),
    )


class ActorCritic(nn.Module):
    """배우와 평론가를 **따로** 둡니다.

    2일차에는 몸통을 공유했는데, 여기서는 나눴습니다. 이유가 있습니다.
    CartPole 의 목표 점수는 100 안팎이라 평론가의 오차가 수천까지 커집니다.
    반면 배우 쪽 손실은 0.1 수준입니다.
    몸통을 공유하면 ==평론가 쪽 기울기가 배우를 밀어내== 학습이 정체합니다.
    (실제로 공유형으로 만들었더니 점수가 130에서 더 안 올랐습니다.)

    따로 두면 서로 간섭하지 않습니다. 파라미터는 조금 늘지만 훨씬 안정적입니다.
    """

    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.actor = mlp(obs_dim, n_actions)   # 행동별 점수 (softmax 전)
        self.critic = mlp(obs_dim, 1)          # 상태 가치

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)

    def act(self, x):
        """행동을 하나 뽑고, 그 확률과 상태 가치를 함께 돌려줍니다."""
        logits, value = self(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), value


def compute_gae(rewards, values, dones, last_value):
    """GAE — 어드밴티지를 계산합니다.

    2일차에 "몇 걸음 뒤에 배울까(n-step)"를 배웠습니다.
    GAE 는 1걸음, 2걸음, 3걸음... 짜리를 전부 계산해서
    가중평균을 내는 방법입니다. LAMBDA 가 그 가중치를 정합니다.
      LAMBDA=0 → 1걸음만 (편향 크고 안정적)
      LAMBDA=1 → 끝까지 (정확하지만 출렁임)
    보통 0.95 근처를 씁니다. n-step 을 일일이 고를 필요가 없어 편합니다.
    """
    adv = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        # (1 - dones[t]) : 판이 끝났으면 미래를 안 봅니다.
        #   2일차 DQN 에서 강조한 그 부분과 같은 역할입니다.
        delta = rewards[t] + GAMMA * next_value * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        adv[t] = gae
    return adv


def main():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    # 학습률을 서서히 줄입니다.
    # 후반에도 크게 움직이면 어렵게 배운 정책이 한 번에 무너집니다.
    # (원 논문도 이 방식을 씁니다. 없으면 점수가 올랐다 떨어졌다 합니다.)
    n_updates = TOTAL_STEPS // ROLLOUT
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.05, total_iters=n_updates)

    state, _ = env.reset(seed=0)
    ep_return, returns_log = 0.0, []
    step_count = 0

    while step_count < TOTAL_STEPS:
        # ── ① 경험 모으기 (ROLLOUT 스텝) ───────────────────
        states, actions, logps, rewards, dones, values = [], [], [], [], [], []

        for _ in range(ROLLOUT):
            s_t = torch.as_tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                a, logp, v = model.act(s_t)

            next_state, r, term, trunc, _ = env.step(a.item())
            done = term or trunc

            states.append(state)
            actions.append(a.item())
            logps.append(logp.item())      # ← 이때의 확률을 기억해 둡니다
            rewards.append(r)
            dones.append(float(term))      # 시간초과(trunc)는 진짜 끝이 아님
            values.append(v.item())

            state = next_state
            ep_return += r
            step_count += 1

            if done:
                returns_log.append(ep_return)
                ep_return = 0.0
                state, _ = env.reset()

        # 마지막 상태의 가치 — 여기서 끊겼을 뿐 아직 남아 있다는 뜻
        with torch.no_grad():
            last_value = model(torch.as_tensor(state, dtype=torch.float32,
                                               device=device))[1].item()

        # ── ② 어드밴티지 계산 ─────────────────────────────
        adv = compute_gae(np.array(rewards), np.array(values),
                          np.array(dones), last_value)
        ret = adv + np.array(values, dtype=np.float32)   # 학습 목표값

        # 어드밴티지를 평균 0, 표준편차 1 로 맞춥니다.
        # 크기가 들쭉날쭉하면 학습이 불안정해집니다.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        S = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
        A = torch.as_tensor(actions, dtype=torch.int64, device=device)
        OLD_LOGP = torch.as_tensor(logps, dtype=torch.float32, device=device)
        ADV = torch.as_tensor(adv, device=device)
        RET = torch.as_tensor(ret, device=device)

        # ── ③ 같은 경험으로 여러 번 학습 ──────────────────
        idx = np.arange(len(S))
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for start in range(0, len(idx), BATCH):
                b = idx[start:start + BATCH]

                logits, value = model(S[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(A[b])

                # 여기가 PPO 의 핵심입니다.
                # ratio = 지금 확률 / 그때 확률
                #   로그끼리 빼고 exp 하면 나눗셈이 됩니다.
                ratio = torch.exp(logp - OLD_LOGP[b])

                # 두 값을 계산해 작은 쪽을 씁니다.
                #   surr1 : 그냥 비율을 곱한 것
                #   surr2 : 비율을 0.8~1.2 로 자른 것
                # 작은 쪽을 쓰면, 정책이 크게 바뀌어도 이득이 늘지 않습니다.
                # 그래서 "크게 바꿀 이유"가 사라집니다.
                surr1 = ratio * ADV[b]
                surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[b]
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.functional.mse_loss(value, RET[b])

                # 엔트로피 — 너무 일찍 한 행동에만 확신하지 말라는 압력.
                # 2일차 A2C 에서 배운 것과 같습니다.
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.25 * critic_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                # 기울기가 너무 크면 잘라 냅니다. 한 번에 확 바뀌는 것을 막습니다.
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        scheduler.step()

        recent = np.mean(returns_log[-20:]) if returns_log else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"스텝 {step_count:6d}  최근 20판 평균 점수 {recent:7.1f}  "
              f"학습률 {cur_lr:.1e}")

        if recent >= 475:
            print("\nCartPole 해결! (475점 이상)")
            break

    env.close()

    # ── 결과 그래프 ─────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
            if _f in {f.name for f in font_manager.fontManager.ttflist}:
                plt.rcParams["font.family"] = _f
                break
        plt.rcParams["axes.unicode_minus"] = False

        plt.figure(figsize=(9, 4))
        plt.plot(returns_log, alpha=0.3, label="판별 점수")
        if len(returns_log) >= 20:
            ma = np.convolve(returns_log, np.ones(20) / 20, mode="valid")
            plt.plot(range(19, len(returns_log)), ma, linewidth=2, label="20판 이동평균")
        plt.axhline(475, linestyle="--", color="gray", label="해결 기준 475")
        plt.xlabel("판 수"); plt.ylabel("점수")
        plt.title("PPO — CartPole 학습 곡선")
        plt.legend(); plt.tight_layout(); plt.show()
    except ImportError:
        pass


if __name__ == "__main__":
    main()


# ══════════════════════════════════════════════════════════
# 해보기
# ══════════════════════════════════════════════════════════
# ① CLIP 을 0.2 → 1.0 으로 바꿔 보세요.
#    사실상 자르지 않는 것이라 A2C 를 여러 번 학습시킨 것과 비슷해집니다.
#    학습이 불안정해지는 것을 볼 수 있습니다.
#
# ② EPOCHS 를 10 → 1 로 바꿔 보세요.
#    경험을 한 번만 쓰는 것이라 A2C 에 가까워집니다.
#    더 많은 경험이 필요해집니다(=느려집니다).
#
# ③ LAMBDA 를 0.95 → 0 으로 바꿔 보세요.
#    1걸음짜리만 쓰는 것이 되어, 안정적이지만 학습이 느려집니다.
#
# ④ 환경을 "LunarLander-v3" 로 바꿔 보세요. (gymnasium[box2d] 필요)
#    상태 8개, 행동 4개라 조금 더 어렵습니다. TOTAL_STEPS 를 늘려야 합니다.
