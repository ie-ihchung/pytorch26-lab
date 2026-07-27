# ============================================================
# [허깅페이스 ②] 알고리즘 3종을 같은 문제에서 나란히 비교
# ------------------------------------------------------------
# 우리가 직접 학습시키려면 알고리즘마다 몇 분씩 걸립니다.
# 이미 학습된 것을 받아 오면 ==몇 초 만에 나란히 비교==할 수 있습니다.
#
# Pendulum-v1 = 막대를 흔들어 거꾸로 세우는 문제 (3일차에 쓰는 환경)
#   · 행동이 연속값입니다 (왼쪽/오른쪽이 아니라 "힘을 얼마나")
#   · 점수는 항상 음수입니다. 0에 가까울수록 잘한 것.
#   · 아무렇게나 하면 -1200 근처가 나옵니다.
#
# ★ 코랩에서 하세요 ★ (수업용 rl 환경에 설치하면 1일차 코드가 깨집니다)
#   !pip install -q stable-baselines3 huggingface_sb3
#
# 실행 시간: 30초
# ============================================================
import numpy as np
import gymnasium as gym
from huggingface_sb3 import load_from_hub
from stable_baselines3 import SAC, TD3

env = gym.make('Pendulum-v1')

# 옛 gym 으로 저장된 모델을 지금 gymnasium 으로 읽기 위한 설정
CUSTOM = {
    'learning_rate': 0.0,
    'lr_schedule': lambda _: 0.0,
    'observation_space': env.observation_space,
    'action_space': env.action_space,
}

MODELS = [
    ('SAC', SAC, 'sb3/sac-Pendulum-v1', 'sac-Pendulum-v1.zip'),
    ('TD3', TD3, 'sb3/td3-Pendulum-v1', 'td3-Pendulum-v1.zip'),
]


def evaluate(model, episodes=5):
    """같은 시드로 몇 판 돌려 평균 점수를 낸다 (공정 비교)."""
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total, done = 0.0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total += reward
            done = term or trunc
        out.append(total)
    return out


print('=' * 58)
print('기준선 — 아무렇게나 했을 때')
print('=' * 58)

rand = []
for ep in range(5):
    obs, _ = env.reset(seed=ep)
    total, done = 0.0, False
    while not done:
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        total += reward
        done = term or trunc
    rand.append(total)
print(f'  무작위      평균 {np.mean(rand):8.1f}')

print()
print('=' * 58)
print('학습된 모델들')
print('=' * 58)

scores = {'random': np.mean(rand)}
for name, cls, repo, fname in MODELS:
    print(f'  {name} 내려받는 중...')
    ckpt = load_from_hub(repo_id=repo, filename=fname)
    model = cls.load(ckpt, custom_objects=CUSTOM)
    rs = evaluate(model)
    scores[name] = np.mean(rs)
    print(f'  {name:10s} 평균 {np.mean(rs):8.1f}   (판별 점수: '
          + ', '.join(f'{v:.0f}' for v in rs) + ')')

print()
print('=' * 58)
print('정리')
print('=' * 58)
for k, v in scores.items():
    bar = '█' * max(1, int((v + 1300) / 40))       # -1300 을 바닥으로 잡은 막대
    print(f'  {k:10s} {v:8.1f}  {bar}')

print('''
  → SAC 와 TD3 는 둘 다 -150 근처입니다. 무작위(-1180)와 비교하면 엄청난 차이죠.
  → 두 알고리즘의 차이는 거의 없습니다. Pendulum 이 쉬운 문제라서 그렇습니다.
    어려운 문제로 갈수록 차이가 벌어집니다.

  ※ 강사 맥북 실측(5판): SAC -141.7, TD3 -145.6, 무작위 -1180.4
    판마다 편차가 큽니다 — TD3 는 -1점부터 -245점까지 나왔습니다.
    막대가 처음에 어느 각도에서 시작하느냐에 따라 난이도가 다르기 때문입니다.
  ※ 3일차에 우리가 직접 만든 SAC 와 비교해 보세요.
    같은 알고리즘인데 학습 시간이 다르면 점수가 얼마나 차이 나는지 볼 수 있습니다.''')

env.close()

# ============================================================
# 더 해보기
#   1) episodes 를 20 으로 늘려 보세요. 평균이 얼마나 흔들리나요?
#   2) deterministic=False 로 두면 SAC 는 확률적으로 행동합니다. 점수 변화는?
#   3) 허깅페이스에서 다른 환경도 찾아보세요:
#        https://huggingface.co/sb3
#      LunarLander, BipedalWalker 등이 있습니다.
#      (다만 Box2D 환경은 코랩에서 !pip install "gymnasium[box2d]" 가 더 필요합니다)
# ============================================================
