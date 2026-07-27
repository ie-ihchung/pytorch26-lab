/**
 * pytorch_basics/*.py 를 읽어 사이트의 src/data/pytorch.js 를 만든다.
 *
 * 왜 생성하는가.
 * 소스를 사이트 데이터 파일에 손으로 옮겨 적으면 두 곳이 어긋난다.
 * 실행해서 검증한 .py 가 정본이고, 사이트는 그 사본이어야 한다.
 * 설명(goal/why/blocks/tryIt)만 여기에 두고 코드는 파일에서 그대로 읽는다.
 *
 *   node scripts/gen_pytorch_page.mjs
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const LAB = path.join(HERE, '..')
const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'

// 백틱을 리터럴로 쓰면 이 파일 자신이 깨진다 (전역 규칙에 기록된 함정)
const BT = String.fromCharCode(96)
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

const SECTIONS = [
  {
    file: '01_tensor.py',
    title: '텐서 — 파이토치의 유일한 재료',
    minutes: 20,
    goal: '숫자를 담는 상자 하나만 이해하면 절반은 끝납니다.',
    why: `파이토치에서 다루는 것은 딱 하나, **텐서(tensor)** 입니다.
엑셀 표를 떠올리시면 됩니다. 숫자 하나면 0차원, 한 줄이면 1차원, 표면 2차원.

==여기서 배울 것 중 가장 중요한 건 "모양(shape)"입니다.==
앞으로 만날 오류의 절반 이상이 모양이 안 맞아서 납니다.
막히면 무조건 \`print(x.shape)\` 부터 찍는 습관을 여기서 들이시면 됩니다.`,
    tryIt: `\`torch.zeros(2,3)\` 의 2와 3을 바꿔 보세요. 출력이 어떻게 달라지나요?
\`v.unsqueeze(0)\` 과 \`v.unsqueeze(1)\` 의 모양 차이를 눈으로 확인해 두세요 — 2일차 코드에 계속 나옵니다.`,
  },
  {
    file: '02_autograd.py',
    title: '자동미분 — 파이토치를 쓰는 진짜 이유',
    minutes: 30,
    goal: '식만 쓰면 미분은 파이토치가 해줍니다. 그게 전부입니다.',
    why: `신경망 학습은 결국 =="어느 쪽으로 조금 움직이면 오차가 줄어드는가"== 를 계속 찾는 일입니다.
그 방향이 미분(기울기, gradient)입니다.

손으로 미분하던 것을 \`backward()\` 한 줄이 대신 해줍니다.
직접 \`y = x²\` 를 미분해서 \`2x\` 가 나오는 것까지 눈으로 확인합니다.

**여기서 나오는 \`zero_grad\` · \`no_grad\` · \`detach\` 세 가지는**
2·3일차 코드에 계속 나옵니다. 여기서 확실히 잡고 가시면 편합니다.`,
    tryIt: `6번 경사하강법에서 \`lr = 0.1\` 을 \`0.5\` 로, 다시 \`0.01\` 로 바꿔 보세요.
너무 크면 튀고 너무 작으면 못 갑니다. 학습률 감각이 여기서 생깁니다.`,
  },
  {
    file: '03_loss.py',
    title: '손실 함수 — "얼마나 틀렸는가"를 숫자 하나로',
    minutes: 25,
    goal: '문제 종류마다 쓰는 자가 다릅니다. 세 가지만 알면 됩니다.',
    why: `학습은 **틀린 정도를 줄이는 일**입니다. 그 틀린 정도를 재는 자가 손실 함수입니다.

숫자를 맞히는 문제면 **MSE**, 분류 문제면 **CrossEntropy**,
강화학습에서 튀는 값이 많으면 **SmoothL1(Huber)** 를 씁니다.

==마지막 4번은 5교시 내용을 미리 보는 것입니다.==
정책 경사 손실에 왜 마이너스가 붙는지, 기울기를 직접 찍어서 확인합니다.`,
    tryIt: `2번에서 차이를 100으로 키워 보세요. MSE는 10000이 되지만 SmoothL1은 99.5입니다.
DQN 코드가 \`smooth_l1_loss\` 를 쓰는 이유가 이 한 줄에 있습니다.`,
  },
  {
    file: '04_linear_regression.py',
    title: '선형회귀 — 학습 5단계를 처음부터 끝까지',
    minutes: 30,
    goal: '앞으로 모든 코드에 그대로 반복될 5줄을 여기서 익힙니다.',
    why: `가장 단순한 학습입니다. 흩어진 점들 사이를 지나는 직선 하나를 찾습니다.

그런데 ==여기 나오는 5단계가 3일 내내 그대로 반복됩니다.==

**① 예측 → ② 손실 → ③ 기울기 지우기 → ④ 역전파 → ⑤ 한 걸음**

DQN도, 정책 경사도, SAC도 이 다섯 줄의 변형일 뿐입니다.
지금 이 5줄을 외워 두시면 뒤가 훨씬 편해집니다.`,
    tryIt: `\`optimizer.zero_grad()\` 줄을 지우고 돌려 보세요. **꼭 한 번 해보시길 권합니다.**
기울기가 쌓여서 엉뚱한 곳으로 가버립니다. 실수했을 때 어떤 증상인지 미리 알아두는 겁니다.`,
  },
  {
    file: '05_nn_module.py',
    title: '신경망 만들기 — nn.Module',
    minutes: 30,
    goal: '층을 쌓는 두 가지 방법과, 배치(batch)라는 개념.',
    why: `직선 하나로는 못 푸는 문제가 대부분이라 층을 여러 개 쌓습니다.

만드는 방법은 두 가지입니다.
**\`nn.Sequential\`** — 위에서 아래로 흘러가기만 하면 이걸로 충분합니다.
**\`class ... (nn.Module)\`** — 중간에 뭔가 해야 하면 직접 만듭니다.

==3번의 "배치" 개념이 특히 중요합니다.==
상태 하나를 넣으면 \`(4,)\`, 32개를 한 번에 넣으면 \`(32, 4)\`.
2일차 코드는 거의 전부 뒤쪽 모양입니다.`,
    tryIt: `\`super().__init__()\` 을 지우고 돌려 보세요. 어떤 오류가 나는지 봐 두시면
나중에 같은 오류를 만났을 때 바로 알아챕니다.`,
  },
  {
    file: '06_activation.py',
    title: '활성화 함수 — 직선을 구부리는 장치',
    minutes: 25,
    goal: '왜 필요한지, 왜 ReLU를 기본으로 쓰는지.',
    why: `\`nn.Linear\` 는 직선입니다. ==직선을 아무리 여러 번 겹쳐도 직선입니다.==
1번 실험에서 층을 3개 쌓아도 결과가 직선인 것을 직접 확인합니다.

그래서 층 사이마다 구부리는 장치를 하나씩 끼웁니다.

3번은 **기울기 소실**을 숫자로 봅니다.
10층을 쌓고 맨 앞 층에 기울기가 얼마나 남았는지 비교하면,
Sigmoid가 왜 밀려나고 ReLU가 왜 기본이 됐는지 한눈에 보입니다.`,
    tryIt: `1번 실험 코드에 \`nn.ReLU()\` 를 끼워 넣고 다시 돌려 보세요.
이웃 차이가 더 이상 일정하지 않게 됩니다 — 그게 "구부러졌다"는 뜻입니다.`,
  },
  {
    file: '07_rl_syntax.py',
    title: '강화학습 코드에서 실제로 만나는 문법만',
    minutes: 30,
    goal: '이 8가지면 2·3일차 코드의 문법은 거의 다 읽힙니다.',
    why: `2·3일차 코드를 읽다 =="이건 뭐지?" 하고 걸리는 것들만== 골라 모았습니다.

\`gather\` · \`max/argmax\` · \`unsqueeze\` · \`no_grad\` · \`detach\` ·
\`Categorical\` · \`Normal+tanh\` · \`stack\`

**3번의 "조용히 잘못되는 경우"를 특히 봐 두세요.**
\`(4,)\` 와 \`(4,1)\` 을 더하면 오류가 안 나고 \`(4,4)\` 로 부풀어 버립니다.
손실이 이상한데 오류는 없을 때, 거의 여기입니다.`,
    tryIt: `1번 gather 예제의 \`actions\` 를 바꿔 가며 어떤 값이 뽑히는지 확인해 보세요.
표에서 자리를 짚어 뽑는 그림이 머리에 들어오면 DQN 코드가 편해집니다.`,
  },
]

const parts = SECTIONS.map((s) => {
  const src = fs.readFileSync(path.join(LAB, 'pytorch_basics', s.file), 'utf8').trimEnd()
  return `  {
    id: '${s.file.replace('.py', '')}',
    title: ${BT}${esc(s.title)}${BT},
    minutes: ${s.minutes},
    goal: ${BT}${esc(s.goal)}${BT},
    why: ${BT}${esc(s.why)}${BT},
    tryIt: ${BT}${esc(s.tryIt)}${BT},
    code: {
      filename: '${s.file}',
      source: ${BT}${esc(src)}${BT},
    },
  },`
}).join('\n')

const out = `// ============================================================
// 파이토치 문법 (2일차 시작 전 · 선택) — 자동 생성 파일
// ------------------------------------------------------------
// 이 파일은 손으로 고치지 마세요. 정본은 pytorch26-lab 리포의
// pytorch_basics/*.py 이며, 전부 실제로 실행해 검증한 코드입니다.
// 고칠 때는 그쪽을 고치고 아래 명령을 다시 돌립니다.
//
//   node scripts/gen_pytorch_page.mjs
//
// 왜 이 페이지가 생겼나 (2026-07-28 대표 지시):
//   2일차 3교시 "PyTorch 소개"를 파이토치 문법 수업으로 알고 오신
//   초보 수강생이 있었다. 강화학습과 별개로 파이토치 자체를 배우려는
//   수요가 분명해서, 교재 vol1(텐서→자동미분→손실→회귀→모듈→활성화)
//   순서 그대로 3시간 분량을 따로 뺐다.
//
// 실습 환경은 코랩(Colab)이 기본이다 — 설치가 필요 없기 때문.
// ============================================================

export default [
${parts}
]
`

fs.writeFileSync(path.join(SITE, 'src/data/pytorch.js'), out, 'utf8')
console.log(`src/data/pytorch.js 생성 — ${SECTIONS.length}개 절, ${out.length.toLocaleString()}자`)
