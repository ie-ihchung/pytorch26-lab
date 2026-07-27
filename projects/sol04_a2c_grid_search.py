# ==========================================================
# 미니 프로젝트 4 [응용] — A2C 하이퍼파라미터 실험실
# 환경: CartPole-v1
# 목표: n-step 길이와 엔트로피 계수가 정책 경사 학습에 주는 영향을 체계적으로 실험한다
# 재사용: 2일차 7교시 A2C 코드에 실험 루프만 추가
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - n_steps ∈ {1, 5, 20} × entropy_coef ∈ {0, 0.01, 0.1} 9개 조합 실험
#  - 각 조합을 시드 3개로 반복해 평균 학습곡선 산출
#  - "n이 길수록 MC에, 짧을수록 TD에 가까워진다"를 곡선으로 확인
#  - 엔트로피 0일 때 조기 수렴(국소 최적) 현상 관찰
# [결과 인증]
#  9개 조합 결과표(최종 평균 리턴) + 최적 조합과 그 이유 한 줄 공유

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import itertools

class AC(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(4, 128), nn.ReLU())
        self.pi = nn.Linear(128, 2)
        self.v = nn.Linear(128, 1)
    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)

def train(n_steps, ent_coef, seed=0, updates=1500):
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    model = AC()
    opt = torch.optim.Adam(model.parameters(), lr=7e-4)
    s, _ = env.reset(seed=seed)
    ep_ret, rets = 0, []
    for _ in range(updates):
        logps, vals, rs, ents, ds = [], [], [], [], []
        for _ in range(n_steps):
            logits, v = model(torch.as_tensor(s, dtype=torch.float32))
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            s2, r, term, trunc, _ = env.step(a.item())
            d = term or trunc
            logps.append(dist.log_prob(a)); vals.append(v)
            rs.append(r); ds.append(d); ents.append(dist.entropy())
            ep_ret += r; s = s2
            if d:
                rets.append(ep_ret); ep_ret = 0
                s, _ = env.reset()
        with torch.no_grad():
            _, v_last = model(torch.as_tensor(s, dtype=torch.float32))
        R, targets = v_last, []
        for r, d in zip(reversed(rs), reversed(ds)):
            R = r + 0.99 * R * (1 - d)
            targets.insert(0, R)
        targets = torch.stack(targets).detach()
        vals = torch.stack(vals)
        adv = targets - vals
        loss = (-(torch.stack(logps) * adv.detach()).mean()
                + 0.5 * adv.pow(2).mean()
                - ent_coef * torch.stack(ents).mean())
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
    return np.mean(rets[-20:]) if len(rets) >= 20 else 0.0

# ── 9개 조합 × 시드 3개 격자 실험 ──
results = {}
for n, e in itertools.product([1, 5, 20], [0.0, 0.01, 0.1]):
    scores = [train(n, e, seed=sd) for sd in range(3)]
    results[(n, e)] = np.mean(scores)
    print(f"n_steps={n:2d}  ent={e:4.2f}  →  평균 {results[(n, e)]:6.1f}")

best = max(results, key=results.get)
print(f"최적 조합: n_steps={best[0]}, ent_coef={best[1]}  ({results[best]:.1f}점)")
# ent=0 조합이 유독 낮다면 조기 수렴(탐험 소멸)을 관찰한 것입니다
