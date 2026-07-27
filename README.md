# pytorch26-lab — 강의용 실습 폴더

**PyTorch로 배우는 강화학습** (2026-07-27~29 · 멀티캠퍼스 역삼) 진행용 실습 코드.

강의 사이트 <https://pytorch26.dreamitbiz.com> 의 코드를 실행 가능한 파일로 뽑아 둔 것이다.

---

## 정본은 사이트다

이 폴더의 `.py` / `.ipynb` 는 **전부 자동 생성물**이다.
정본은 사이트 리포의 `src/data/*.js` 이고, 여기 파일을 직접 고쳐도 **다음 추출 때 덮어써진다.**

코드를 고칠 때는 사이트를 고치고 다시 뽑는다.

```bash
node scripts/extract-lab.mjs
```

VS Code에서는 `Cmd+Shift+P` → `Tasks: Run Task` → **🔄 사이트에서 실습코드 다시 추출**.

---

## 폴더 구성

```
day1/ day2/ day3/     교시별 코드 (일차당 7교시 = 21개) + dayN_전체.ipynb
projects/             미니 프로젝트 완성 소스 8개 (sol01~sol08)
student-setup/        수강생 배포용 설치 안내·점검 스크립트
scripts/              추출기
00_env_check.py       환경 점검 (사이트 환경설정 페이지 정본)
```

---

## 교시별 `.py` 가 단독으로 안 도는 것은 정상이다

사이트 `day1.js` 주석에 적힌 대로, 수업은 **하나의 파일에 코드를 이어 붙이며** 진행하는 전제로 짜여 있다.

- 1일차 3~5교시는 2교시의 `GridWorld`(`P`, `n_states`, `n_actions`)를 그대로 이어 쓴다
- 2일차 4교시는 앞 교시의 `QNetwork`, 7교시는 `ActorCritic` 을 이어 쓴다
- 3일차 2교시는 `Actor`, 6교시는 `GaussianActor` 를 이어 쓴다

그래서 `03_policy_evaluation.py` 를 단독 실행하면 `NameError: name 'n_states' is not defined` 가 난다 — **버그가 아니다.**

**강의 진행은 `dayN_전체.ipynb` 로 한다.** 교시 순서대로 셀이 배치돼 있고 위에서부터 실행하면 이어진다.
교시별 `.py` 는 특정 교시 코드만 화면에 크게 띄우거나 수강생에게 파일로 나눠 줄 때 쓴다.

미니 프로젝트 `sol05` · `sol06` 도 같은 이유로 단독 실행되지 않는다 — 3일차 수업 코드(클래스)를 전제한 러너다.
`sol03`(LunarLander)은 Box2D가 있어야 한다.

---

## 실습환경

conda 환경 `rl` (Python 3.10) 을 쓴다. VS Code 인터프리터는 `.vscode/settings.json` 에 고정해 뒀다.

```bash
conda activate rl
python 00_env_check.py
```

### 이 맥(Intel)에서 밟은 함정 — NumPy 버전

이 맥은 Intel(x86_64)이라 **PyTorch가 2.2.2에서 멈춰 있다** (그 이후 버전은 x86 macOS 휠을 내지 않는다).
2.2.2는 NumPy 1.x에 맞춰 빌드된 것이라, `pip install torch numpy` 로 최신 NumPy 2.x가 들어가면

```
RuntimeError: Numpy is not available
```

가 나면서 `torch.from_numpy()` 가 죽는다. 강화학습 코드는 매 스텝 관측(NumPy)을 텐서로 바꾸므로 **실습이 첫 줄부터 막힌다.**
`import torch` 자체는 성공하고 경고만 뜨기 때문에 알아채기 어렵다.

→ 이 환경은 `numpy<2` 로 고정해 뒀다. 재설치할 일이 있으면 반드시 같이 맞출 것.

**수강생 Windows PC는 해당 없다** — 최신 torch가 설치되므로 NumPy 2.x와 정상 동작한다.

### 수강생 환경과 다른 점

| | 강사 (이 맥) | 수강생 (실습 PC) |
|---|---|---|
| CPU/GPU | Intel i9-9880H · GPU 없음 | i7-8700 · RTX 2070 |
| torch | 2.2.2 (x86 macOS 마지막 빌드) | 최신 + CUDA 12.6 빌드 |
| numpy | 1.26.4 로 고정 필수 | 제약 없음 |
| 설치 명령 | `pip install torch` | `pip install torch --index-url .../cu126` |

수강생 쪽에서 `--index-url` 을 빼면 **CPU 전용 휠(122MB)** 이 깔려 RTX 2070을 못 쓴다.
자세한 안내는 `student-setup/수강생_실습환경_안내.md`.

---

## VS Code

`01-edu-sites/pytorch26.code-workspace` 를 열면 이 폴더와 강의 사이트가 한 창에 뜬다.

| 단축 | 동작 |
|---|---|
| `F5` | 현재 실습 파일 실행 |
| `Cmd+Shift+P` → Tasks | 환경점검 / 전체 문법검사 / 재추출 / 사이트 로컬 실행 |

강의용으로 인라인 제안(Copilot 류)과 Pylance 타입경고를 꺼 뒀다 — 수강생이 내 타이핑과 헷갈리고, 빨간 줄이 늘면 설명이 산만해지기 때문이다.
