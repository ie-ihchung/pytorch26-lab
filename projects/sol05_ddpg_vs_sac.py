# ==========================================================
# 미니 프로젝트 5 [응용] — Pendulum — DDPG vs SAC 대결
# 환경: Pendulum-v1
# 목표: 같은 연속 제어 문제에서 결정적 정책과 최대 엔트로피 정책의 안정성을 비교한다
# 재사용: 3일차 2교시 DDPG + 6교시 SAC 코드 그대로
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - 두 알고리즘을 같은 시드 3개로 각각 학습 (에피소드 150)
#  - 평균±표준편차 밴드가 있는 학습곡선을 겹쳐 그리기
#  - DDPG의 탐험 노이즈 σ를 0.05/0.1/0.3으로 바꿔 민감도 확인
#  - SAC의 α(온도)가 학습 중 어떻게 변하는지 로그 찍어 관찰
# [결과 인증]
#  두 알고리즘 학습곡선 비교(시드 3개 평균) + 어느 쪽이 안정적이었는지 관찰 공유

# 전제: 3일차 실습 코드(Actor, Critic, GaussianActor, ReplayBuffer,
#        soft_update)가 같은 노트북/파일에 이미 정의되어 있음
# 아래는 두 알고리즘을 "함수화 → 시드 3개 → 밴드 그래프"로 비교하는 러너
import gymnasium as gym
import torch, copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 그래프 한글 깨짐 방지 — 설치된 한글 폰트를 자동으로 골라 씁니다.
# Windows는 맑은 고딕, macOS는 AppleGothic이 기본 탑재입니다.
# Colab 등 리눅스는 !apt -qq install fonts-nanum 후 런타임을 다시 시작하세요.
for _f in ("Malgun Gothic", "AppleGothic", "NanumGothic", "NanumBarunGothic"):
    if _f in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False

def run_ddpg(seed, episodes=150, noise_std=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    env = gym.make("Pendulum-v1")
    actor, critic = Actor(3, 1, 2.0), Critic(3, 1)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    a_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
    c_opt = torch.optim.Adam(critic.parameters(), lr=1e-3)
    buf = ReplayBuffer(100_000)
    rets = []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 1000 + ep)
        done, tot = False, 0.0
        while not done:
            with torch.no_grad():
                a = actor(torch.as_tensor(s, dtype=torch.float32)).numpy()
            a = (a + np.random.normal(0, noise_std, 1)).clip(-2, 2)
            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.push(s, a, r, s2, float(term))
            s, tot = s2, tot + r
            if len(buf) >= 1000:
                bs, ba, br, bs2, bd = buf.sample(128)
                with torch.no_grad():
                    y = br + 0.99 * critic_t(bs2, actor_t(bs2)) * (1 - bd)
                c_loss = torch.nn.functional.mse_loss(critic(bs, ba), y)
                c_opt.zero_grad(); c_loss.backward(); c_opt.step()
                a_loss = -critic(bs, actor(bs)).mean()
                a_opt.zero_grad(); a_loss.backward(); a_opt.step()
                soft_update(actor_t, actor); soft_update(critic_t, critic)
        rets.append(tot)
    return rets

def run_sac(seed, episodes=150):
    # 6교시 SAC 코드(트윈 Q + 자동 alpha)를 함수로 감싸 rets를 반환하게
    # 만드세요 — 셋업 블록과 학습 루프를 그대로 옮기면 됩니다.
    raise NotImplementedError("6교시 SAC 코드를 함수로 감싸 넣으세요")

def band_plot(all_runs, label):
    arr = np.array(all_runs)             # (시드 3, 에피소드)
    m, sd = arr.mean(0), arr.std(0)
    x = np.arange(arr.shape[1])
    plt.plot(x, m, label=label)
    plt.fill_between(x, m - sd, m + sd, alpha=0.2)

band_plot([run_ddpg(sd) for sd in range(3)], "DDPG")
band_plot([run_sac(sd) for sd in range(3)], "SAC")
plt.xlabel("episode"); plt.ylabel("return")
plt.legend(); plt.title("Pendulum-v1: DDPG vs SAC (seed 3개 평균±표준편차)")
plt.show()
# 관찰 포인트: SAC 밴드(표준편차)가 좁게 유지되는가?
# noise_std를 0.05/0.1/0.3으로 바꿔 DDPG 민감도도 확인해 보세요
