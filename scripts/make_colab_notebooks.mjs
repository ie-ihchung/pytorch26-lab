/**
 * pytorch_basics/*.py 를 코랩용 .ipynb 로 만든다.
 *
 * 왜 필요한가 (2026-07-28 대표 지시):
 *   "코랩 파일은 설명도 하지만 바로 코랩에서 열어서 할 수 있게"
 *   사이트에서 복사·붙여넣기를 시키면 초보자가 한 칸에 다 넣고 실행하다
 *   어디서 틀렸는지 못 찾는다. 노트북으로 주면 셀이 이미 나뉘어 있다.
 *
 * 셀 나누는 기준:
 *   .py 안의  print('=' * NN)  구분선을 절 경계로 본다.
 *   구분선 바로 앞 제목 print 를 마크다운 셀로 올린다.
 *
 * 출력: <OUT_DIR>/<이름>.ipynb
 *   node scripts/make_colab_notebooks.mjs <OUT_DIR>
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const LAB = path.join(HERE, '..')
const OUT_DIR = process.argv[2]
if (!OUT_DIR) { console.error('사용법: node scripts/make_colab_notebooks.mjs <OUT_DIR>'); process.exit(1) }
fs.mkdirSync(OUT_DIR, { recursive: true })

// 절별 제목·학습목표·결괏값 해설 (사이트와 같은 내용을 노트북에도 넣는다)
const META = {
  '01_tensor': {
    title: '① 텐서 — 파이토치의 유일한 재료',
    minutes: 25,
    goals: [
      '텐서가 무엇인지, 리스트·넘파이와 어떻게 다른지 말할 수 있다',
      '모양(shape)과 자료형(dtype)을 확인하고 바꿀 수 있다',
      'reshape · unsqueeze · squeeze 로 모양을 맞출 수 있다',
      '자주 나는 오류 두 가지를 스스로 고칠 수 있다',
    ],
    intro: `파이토치에서 다루는 것은 딱 하나, **텐서(tensor)** 입니다.
엑셀 표를 떠올리시면 됩니다. 숫자 하나면 0차원, 한 줄이면 1차원, 표면 2차원입니다.

넘파이 배열과 거의 같은데 두 가지가 다릅니다.
**① GPU 로 옮길 수 있습니다.  ② 미분을 자동으로 해줍니다.**

앞으로 만날 오류의 절반 이상이 **모양(shape)이 안 맞아서** 납니다.
막히면 무조건 \`print(x.shape)\` 부터 찍는 습관을 여기서 들이세요.`,
    outro: `### 결괏값 해설

- \`tensor([1., 2., 3.])\` — 숫자 뒤에 **점**이 붙은 것이 소수(float)라는 뜻입니다.
- 소수점을 안 찍은 \`b\` 는 \`int64\` — **정수**가 됩니다. 신경망에 넣으면 오류가 납니다.
- 같은 12개 숫자가 \`[12]\` → \`[3, 4]\` 로 바뀝니다. 개수는 그대로입니다.
- \`unsqueeze(0)\` 은 앞에, \`unsqueeze(1)\` 은 뒤에 1을 끼워 넣습니다.
- \`p * q\` 는 자리끼리 곱, \`p @ q\` 는 다 곱해서 더함. **둘은 완전히 다릅니다.**

### 직접 해보기
\`torch.zeros(2, 3)\` 의 2와 3을 바꿔 보세요. 어느 쪽이 줄 수인가요?`,
  },
  '02_autograd': {
    title: '② 자동미분 — 파이토치를 쓰는 진짜 이유',
    minutes: 30,
    goals: [
      'backward() 가 무엇을 계산하는지 손으로 검산할 수 있다',
      'zero_grad() 를 왜 매번 부르는지 설명할 수 있다',
      'no_grad 와 detach 의 차이를 구분해 쓸 수 있다',
    ],
    intro: `신경망 학습은 결국 **"어느 쪽으로 조금 움직이면 오차가 줄어드는가"** 를 찾는 일입니다.
그 방향이 미분(기울기)입니다.

손으로 미분하던 것을 \`backward()\` 한 줄이 대신 해줍니다.
이게 파이토치를 쓰는 가장 큰 이유입니다.

여기 나오는 \`zero_grad\` · \`no_grad\` · \`detach\` 세 가지는 **오후 실습 코드에 계속 나옵니다.**`,
    outro: `### 결괏값 해설

- \`x.grad = 6.0\` — 손으로 계산한 \`2x = 2×3 = 6\` 과 같습니다. 마법이 아닙니다.
- **3번이 제일 중요합니다.** backward 를 세 번 부르니 grad 가 **2 → 4 → 6** 으로 쌓입니다.
  그래서 학습 루프마다 \`zero_grad()\` 를 부릅니다.
- 5번 — detach 한 쪽은 \`None\`. **아무것도 안 흘렀다는 뜻**입니다.
  오후 Actor-Critic 에서 이걸 빠뜨리면 학습이 조용히 망가집니다.
- 6번 — x 가 0 에서 3.859 까지 갑니다. 정답 4에 다가갑니다.

### 직접 해보기
마지막 셀의 \`lr = 0.1\` 을 \`0.5\` 로, 다시 \`0.01\` 로 바꿔 보세요.
너무 크면 튀고, 너무 작으면 못 갑니다. **학습률 감각이 여기서 생깁니다.**`,
  },
  '03_loss': {
    title: '③ 손실 함수 — "얼마나 틀렸는가"를 숫자 하나로',
    minutes: 25,
    goals: [
      '문제 종류에 따라 MSE · CrossEntropy · SmoothL1 을 골라 쓸 수 있다',
      'SmoothL1 을 강화학습에서 쓰는 이유를 설명할 수 있다',
    ],
    intro: `학습은 **틀린 정도를 줄이는 일**입니다. 그 틀린 정도를 재는 자가 손실 함수입니다.

- 숫자를 맞히는 문제 → **MSE**
- 분류하는 문제 → **CrossEntropy**
- 값이 가끔 크게 튀는 강화학습 → **SmoothL1 (Huber)**

마지막 4번은 오후 내용을 미리 보는 것입니다. 다 이해 안 되셔도 됩니다.`,
    outro: `### 결괏값 해설

- 차이 \`[0, 1, -3]\` → 제곱 \`[0, 1, 9]\` → 평균 \`3.333\`. 파이토치 결과와 같습니다.
- **차이가 10 일 때 MSE 는 100, SmoothL1 은 9.5.**
  제곱은 크게 틀린 하나에 학습이 통째로 휘둘립니다.
  오후 DQN 이 \`smooth_l1_loss\` 를 쓰는 이유가 이 두 줄에 있습니다.
- 4번 — G가 **+10 일 때와 -10 일 때 기울기 부호가 반대**입니다.
  잘 됐으면 그 행동의 확률을 올리고, 안 됐으면 내립니다.

### 직접 해보기
2번에서 차이를 100 으로 키워 보세요. MSE 10000, SmoothL1 99.5 입니다.`,
  },
  '04_linear_regression': {
    title: '④ 선형회귀 — 학습 5단계를 처음부터 끝까지',
    minutes: 30,
    goals: [
      '학습 5단계를 순서대로 쓸 수 있다',
      'zero_grad() 를 빼면 무슨 일이 나는지 직접 확인한다',
    ],
    intro: `가장 단순한 학습입니다. 흩어진 점들 사이를 지나는 직선 하나를 찾습니다.

그런데 **여기 나오는 5단계가 3일 내내 그대로 반복됩니다.**

> ① 예측 → ② 손실 → ③ 기울기 지우기 → ④ 역전파 → ⑤ 한 걸음

오후에 만들 DQN 도, 내일 만들 SAC 도 이 다섯 줄의 변형일 뿐입니다.
**지금 이 5줄만 외워 두시면 오후가 훨씬 편해집니다.**`,
    outro: `### 결괏값 해설

- 처음 w, b 는 아무 숫자입니다. **신경망은 무작위에서 출발합니다.**
- epoch 이 늘수록 w 는 3 쪽, b 는 2 쪽으로 갑니다.
  데이터를 만들 때 쓴 정답이 \`y = 3x + 2\` 였습니다. **찾아낸 겁니다.**
- **x=5 는 학습에 없던 값**입니다(-3~3 만 배웠음). 그런데도 17 근처를 맞힙니다.
  직선이라는 구조를 배웠기 때문입니다. **1일차 표였다면 못 했을 일입니다.**

### 직접 해보기 (꼭 해보세요)
\`optimizer.zero_grad()\` 줄을 지우고 돌려 보세요.
기울기가 쌓여 엉뚱한 곳으로 갑니다. **실수했을 때 증상을 미리 봐 두는 겁니다.**`,
  },
  '05_nn_module': {
    title: '⑤ 신경망 만들기 — nn.Module',
    minutes: 30,
    goals: [
      'nn.Sequential 과 class 방식을 구분해 쓸 수 있다',
      '배치(batch)가 무엇인지 모양으로 설명할 수 있다',
    ],
    intro: `직선 하나로는 못 푸는 문제가 대부분이라 층을 여러 개 쌓습니다.

- **\`nn.Sequential\`** — 위에서 아래로 흘러가기만 하면 이걸로 충분
- **\`class ... (nn.Module)\`** — 중간에 뭔가 해야 하면 직접 만듦

**3번의 "배치" 개념이 특히 중요합니다.** 오후 코드는 거의 전부 그 모양입니다.`,
    outro: `### 결괏값 해설

- \`Linear(in_features=4, out_features=64)\` = "4개 받아 64개 내보냄"
- **하나를 넣으면 \`[2]\`, 32개를 넣으면 \`[32, 2]\`.** 앞의 32 가 배치입니다.
- 파라미터 개수 합계 — 이 숫자들을 조금씩 고치는 것이 학습입니다.
- 저장했다 불러온 모델이 **원본과 똑같은 값**을 냅니다 (True).

### 직접 해보기
\`super().__init__()\` 을 지우고 돌려 보세요. 어떤 오류가 나는지 봐 두시면
나중에 같은 오류를 만났을 때 바로 알아챕니다.`,
  },
  '06_activation': {
    title: '⑥ 활성화 함수 — 직선을 구부리는 장치',
    minutes: 25,
    goals: [
      '활성화 함수가 없으면 왜 층을 쌓아도 소용없는지 확인한다',
      '강화학습 신경망의 마지막 층에 활성화를 안 붙이는 이유를 안다',
    ],
    intro: `\`nn.Linear\` 는 직선입니다. **직선을 아무리 여러 번 겹쳐도 직선입니다.**
1번에서 층을 3개 쌓아도 결과가 직선인 것을 **숫자로** 확인합니다.

3번은 **기울기 소실**을 숫자로 봅니다.
Sigmoid 가 왜 밀려나고 ReLU 가 왜 기본이 됐는지 한눈에 보입니다.`,
    outro: `### 결괏값 해설

- 1번 — 이웃 차이가 **전부 같은 숫자**입니다. 층을 3개 쌓아도 그냥 직선이라는 뜻입니다.
- 3번 — 10층 통과 후 맨 앞 층의 기울기가
  Sigmoid 는 \`e-10\` 대, ReLU 는 \`e-02\` 대. **억 배 차이입니다.**
  Sigmoid 는 앞쪽 층이 배우지를 못합니다. 이게 기울기 소실입니다.

### 직접 해보기
1번 코드에 \`nn.ReLU()\` 를 끼워 넣고 다시 돌려 보세요.
이웃 차이가 더 이상 일정하지 않게 됩니다 — 그게 "구부러졌다"는 뜻입니다.`,
  },
  '07_rl_syntax': {
    title: '⑦ 오후 실습에 나오는 문법만 모아서',
    minutes: 30,
    goals: [
      'gather 로 "실제로 한 행동의 값"을 뽑을 수 있다',
      '모양이 안 맞아 조용히 틀리는 경우를 알아챌 수 있다',
    ],
    intro: `오후 실습 코드를 읽다 **"이건 뭐지?"** 하고 걸리는 것들만 골라 모았습니다.

\`gather\` · \`max/argmax\` · \`unsqueeze\` · \`no_grad\` · \`detach\` · \`Categorical\` · \`Normal+tanh\` · \`stack\`

**3번의 "조용히 잘못되는 경우"를 특히 봐 두세요.**`,
    outro: `### 결괏값 해설

- 1번 — 4×3 표에서 \`[1, 0, 2, 1]\` 번째를 뽑으니 \`5, 7, 9, 6\`.
  **오후 DQN 이 "내가 한 행동의 Q값"을 뽑는 방법이 이겁니다.**
- **3번이 가장 조심할 곳입니다.** 모양을 잘못 섞으면 \`[4]\` 가 \`[4, 4]\` 가 됩니다.
  **오류가 안 납니다.** 조용히 16개짜리가 만들어져 엉뚱한 손실이 계산됩니다.
- 5번 — 균등할 때 엔트로피가 \`1.0986\` = ln3. 골고루일수록 큽니다.

### 직접 해보기
1번의 \`actions\` 를 바꿔 가며 어떤 값이 뽑히는지 확인해 보세요.`,
  },
}

const mdCell = (text) => ({
  cell_type: 'markdown',
  metadata: {},
  source: text.split('\n').map((l, i, a) => (i === a.length - 1 ? l : l + '\n')),
})
const codeCell = (text) => ({
  cell_type: 'code',
  metadata: {},
  execution_count: null,
  outputs: [],
  source: text.split('\n').map((l, i, a) => (i === a.length - 1 ? l : l + '\n')),
})

/**
 * .py 를 셀 단위로 쪼갠다.
 *
 * 파일 안의 절 구분은 이 모양이다:
 *     print()
 *     print('=' * 58)
 *     print('1. 텐서 만들기 ...')
 *     print('=' * 58)
 * 그래서 "print() 다음 줄이 구분선"인 자리에서 끊는다.
 * (앞서 빈 줄 기준으로 끊으려 했더니 한 셀로 뭉쳐 나왔다 — 초보자가
 *  한 칸에 다 돌리면 어디서 틀렸는지 못 찾으므로 반드시 나눠야 한다)
 */
function splitCells(src) {
  const lines = src.split('\n')
  const isRule = (l) => /^print\('=' \* \d+\)$/.test((l ?? '').trim())
  const chunks = []
  let cur = []

  const hasCode = (arr) => arr.some((l) => l.trim() && !l.trim().startsWith('#'))

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const startsSection = line.trim() === 'print()' && isRule(lines[i + 1])
    if (startsSection && hasCode(cur)) {
      while (cur.length && cur[cur.length - 1].trim() === '') cur.pop()
      chunks.push(cur.join('\n'))
      cur = []
      continue                     // print() 한 줄은 버린다 (셀 첫 줄 빈 출력 방지)
    }
    cur.push(line)
  }
  while (cur.length && cur[cur.length - 1].trim() === '') cur.pop()
  if (hasCode(cur)) chunks.push(cur.join('\n'))
  return chunks
}

let made = 0
for (const [id, meta] of Object.entries(META)) {
  const src = fs.readFileSync(path.join(LAB, 'pytorch_basics', `${id}.py`), 'utf8').trimEnd()
  const chunks = splitCells(src)

  const cells = []

  // 표지
  cells.push(mdCell(`# ${meta.title}

**PyTorch로 배우는 강화학습 · 2일차 오전 · 약 ${meta.minutes}분**

이애본 · DreamIT Biz · https://pytorch26.dreamitbiz.com

---

## 🎯 학습목표

${meta.goals.map((g) => `- ${g}`).join('\n')}

---

## 이 절은

${meta.intro}

---

### 사용법
셀을 **위에서부터 하나씩** \`Shift + Enter\` 로 실행하세요.
한 칸에 몰아서 실행하지 마시고, **한 셀 돌리고 결과 보고** 다음으로 넘어가시면 됩니다.
설치할 것은 없습니다. 파이토치는 코랩에 이미 들어 있습니다.`))

  // 코드 셀들
  chunks.forEach((c) => cells.push(codeCell(c)))

  // 마무리
  cells.push(mdCell(`---

${meta.outro}

---

막히면 사이트의 같은 절을 보세요 — 실행 결과가 그대로 실려 있습니다.
https://pytorch26.dreamitbiz.com/#/day/2/pytorch`))

  const nb = {
    nbformat: 4,
    nbformat_minor: 0,
    metadata: {
      colab: { provenance: [], toc_visible: true },
      kernelspec: { name: 'python3', display_name: 'Python 3' },
      language_info: { name: 'python' },
    },
    cells,
  }

  const out = path.join(OUT_DIR, `${id}.ipynb`)
  fs.writeFileSync(out, JSON.stringify(nb, null, 1), 'utf8')
  console.log(`  ${id}.ipynb — 셀 ${cells.length}개 (코드 ${chunks.length})`)
  made++
}

console.log(`\n노트북 ${made}개 생성 → ${OUT_DIR}`)
