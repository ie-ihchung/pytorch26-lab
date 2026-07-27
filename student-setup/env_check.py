# ============================================================
#  실습환경 점검 — PyTorch로 배우는 강화학습 (2026-07-27~29)
#  이 파일을 실행해서 아래가 전부 [OK]로 나오면 준비 완료입니다.
#      python env_check.py
# ============================================================
import sys
import platform

ok = True


def check(label, fn):
    global ok
    try:
        print(f"[OK]   {label}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[실패] {label}: {type(e).__name__} — {e}")


print("=" * 56)
print(f" 운영체제 : {platform.system()} {platform.release()}")
print(f" 파이썬   : {sys.version.split()[0]}")
print("=" * 56)

if sys.version_info < (3, 10):
    ok = False
    print("[실패] 파이썬이 3.10보다 낮습니다. conda create -n rl python=3.10 으로 다시 만들어 주세요.")

check("PyTorch", lambda: __import__("torch").__version__)
check("Gymnasium", lambda: __import__("gymnasium").__version__)
check("NumPy", lambda: __import__("numpy").__version__)
check("Matplotlib", lambda: __import__("matplotlib").__version__)

# NumPy ↔ PyTorch 연동 — 강화학습은 매 스텝 관측(NumPy)을 텐서로 바꾸므로
# 이 항목이 실패하면 실습이 첫 줄부터 막힌다. torch 빌드와 NumPy 버전이
# 어긋날 때 발생하며, 그때는 pip install "numpy<2" 로 맞춘다.
def numpy_bridge():
    import numpy as np
    import torch

    t = torch.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    return f"torch.from_numpy 정상 (합 {t.sum().item():.0f})"


check("NumPy↔PyTorch 연동", numpy_bridge)


# GPU — 없어도 3일 과정은 CPU로 전부 진행 가능하다. 있으면 2·3일차가 빨라진다.
def gpu():
    import torch

    if torch.cuda.is_available():
        return f"사용 가능 — {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    return "사용 불가 — CPU로 진행합니다 (이번 과정은 CPU로 충분합니다)"


check("GPU(CUDA)", gpu)


# 3일간 실제로 쓰는 환경 전부를 미리 만들어 본다.
# LunarLander는 Box2D가 있어야 하므로 여기서 걸러진다.
def envs():
    import gymnasium as gym

    ids = [
        "CartPole-v1",
        "Pendulum-v1",
        "FrozenLake-v1",
        "CliffWalking-v1",
        "Taxi-v4",
        "MountainCar-v0",
        "LunarLander-v3",
    ]
    bad = []
    for i in ids:
        try:
            e = gym.make(i)
            e.reset(seed=0)
            e.close()
        except Exception as ex:
            bad.append(f"{i}({type(ex).__name__})")
    if bad:
        raise RuntimeError("사용 불가: " + ", ".join(bad))
    return f"{len(ids)}종 전부 정상"


check("실습 환경(Gym)", envs)

print("=" * 56)
if ok:
    print(" 준비 완료 — 이대로 수업에 들어오시면 됩니다.")
else:
    print(" 실패 항목이 있습니다.")
    print(" 위 화면을 그대로 캡처해서 강사에게 보내주세요.")
    print(" 설치가 끝내 안 되면 Google Colab으로도 전체 실습이 가능합니다.")
print("=" * 56)
