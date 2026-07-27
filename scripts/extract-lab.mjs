// pytorch26 사이트 데이터 → 강의용 실습 .py 파일로 추출
// 정본은 사이트(src/data/*.js). 이 스크립트는 재실행 안전(덮어쓰기).
import { mkdirSync, writeFileSync, rmSync } from 'node:fs'
import { join } from 'node:path'

const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'
const LAB = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26-lab'

const { days, course } = await import(join(SITE, 'src/data/curriculum.js'))
const projects = (await import(join(SITE, 'src/data/projects.js'))).default
const setup = (await import(join(SITE, 'src/data/setup.js'))).default

// 형광펜(==...==)·볼드(**...**) 마크업을 평문으로
const plain = (s) => String(s ?? '').replace(/==(.+?)==/g, '$1').replace(/\*\*(.+?)\*\*/g, '$1')

const pad2 = (n) => String(n).padStart(2, '0')
let stats = { day: 0, proj: 0 }

// 일차 노트북 맨 앞에 실어 줄 "앞 일차 재사용 코드" — { 대상일차: [{day, slot}] }
const CARRY = { 3: [{ day: 2, slot: 1 }] }

// 실측 실행 시간(초) — 2026-07-26 강사 맥(Intel i9-9880H, CPU) 기준.
// 학습 루프가 긴 교시는 강의 중 화면이 그만큼 멈추므로, 설명 분량을 미리
// 배분할 수 있게 노트북에 표기한다. 대표 결정으로 에피소드 수는 줄이지 않는다.
// 값을 갱신하려면 `python scripts/verify-run.py`로 다시 측정할 것.
const RUNTIME = {
  '2-4': 52, '2-5': 77, '2-7': 16,
  '3-2': 360, '3-6': 523,
}
const runtimeNote = (dayId, slot) => {
  const s = RUNTIME[`${dayId}-${slot}`]
  if (!s) return null
  const m = Math.floor(s / 60)
  return m >= 1
    ? `⏳ 실행에 약 ${m}분 ${s % 60}초 걸립니다 (학습 루프 — 멈춘 것이 아닙니다)`
    : `⏳ 실행에 약 ${s}초 걸립니다`
}

// ── 일차별 교시 코드 ────────────────────────────────────────
for (const day of days) {
  const dir = join(LAB, `day${day.id}`)
  rmSync(dir, { recursive: true, force: true })
  mkdirSync(dir, { recursive: true })

  for (const s of day.sessions) {
    if (!s.code) continue
    const header = [
      '# ' + '='.repeat(58),
      `# ${day.id}일차 ${s.slot}교시 — ${s.title}`,
      `# ${day.date} ${s.time} · ${day.theme}`,
      `# 원본 파일명: ${s.code.filename}`,
      `# 사이트: https://pytorch26.dreamitbiz.com/#/day/${day.id}#s${s.slot}`,
      '# ' + '='.repeat(58),
      ...(s.objectives?.length
        ? ['# [학습목표]', ...s.objectives.map((o) => `#  - ${plain(o)}`)]
        : []),
      '#',
      `# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.`,
      `#   단독 실행 시 NameError가 나면 정상입니다 — day${day.id}_전체.ipynb를 위에서부터 실행하세요.`,
      '',
      '',
    ].join('\n')
    const name = `${pad2(s.slot)}_${s.code.filename}`
    writeFileSync(join(dir, name), header + s.code.source.trimEnd() + '\n', 'utf8')
    stats.day++
  }
}

// ── 일차별 누적 노트북 (.ipynb) ─────────────────────────────
// 사이트 day1.js 주석대로 3~5교시는 2교시의 GridWorld를, 2·3일차는
// 앞 교시의 클래스를 그대로 이어 쓴다. 따라서 교시 파일을 단독 실행하면
// NameError가 나는 것이 정상이다 — 수업은 "하나의 노트북에 이어 붙이며"
// 진행하는 전제이므로, 그 전제를 그대로 구현한 노트북을 함께 만든다.
for (const day of days) {
  const cells = [
    {
      cell_type: 'markdown',
      id: `d${day.id}-intro`,
      metadata: {},
      source: [
        `# ${day.id}일차 — ${day.theme}\n`,
        `\n`,
        `${day.date} · ${plain(day.desc)}\n`,
        `\n`,
        `> 위에서부터 순서대로 실행하세요. 뒤 교시가 앞 교시의 변수·클래스를 그대로 이어 씁니다.\n`,
      ],
    },
  ]
  // 3일차 DDPG/SAC 코드는 2일차 1교시의 ReplayBuffer를 그대로 쓴다
  // ("2일차의 복선 회수" — day3.js에 명시된 의도된 설계).
  // 3일차 아침에 노트북을 새로 열면 NameError가 나므로, 재사용 코드를
  // 맨 앞 셀로 실어 준다. 사이트 콘텐츠(강의 서사)는 건드리지 않는다.
  for (const c of CARRY[day.id] ?? []) {
    const src = days.find((d) => d.id === c.day)?.sessions.find((s) => s.slot === c.slot)
    if (!src?.code) continue
    cells.push({
      cell_type: 'markdown',
      id: `d${day.id}-carry-${c.day}${c.slot}`,
      metadata: {},
      source: [
        `## (이어받기) ${c.day}일차 ${c.slot}교시 — ${src.title}\n`,
        `\n`,
        `오늘 코드가 이 정의를 그대로 이어 씁니다. **가장 먼저 한 번 실행**하세요.\n`,
      ],
    })
    cells.push({
      cell_type: 'code',
      id: `d${day.id}-carry-${c.day}${c.slot}-code`,
      execution_count: null,
      metadata: {},
      outputs: [],
      source: src.code.source.trimEnd().split('\n').map((l, i, a) => (i === a.length - 1 ? l : l + '\n')),
    })
  }

  for (const s of day.sessions) {
    if (!s.code) continue
    cells.push({
      cell_type: 'markdown',
      id: `d${day.id}-s${s.slot}-md`,
      metadata: {},
      source: [
        `## ${s.slot}교시 · ${s.title}\n`,
        `\n`,
        `\`${s.time}\` · \`${s.code.filename}\`\n`,
        ...(runtimeNote(day.id, s.slot) ? ['\n', `> ${runtimeNote(day.id, s.slot)}\n`] : []),
        ...(s.objectives?.length ? ['\n', ...s.objectives.map((o) => `- ${plain(o)}\n`)] : []),
      ],
    })
    cells.push({
      cell_type: 'code',
      id: `d${day.id}-s${s.slot}-code`,
      execution_count: null,
      metadata: {},
      outputs: [],
      source: s.code.source.trimEnd().split('\n').map((l, i, a) => (i === a.length - 1 ? l : l + '\n')),
    })
  }
  const nb = {
    cells,
    metadata: {
      kernelspec: { display_name: 'Python 3 (rl)', language: 'python', name: 'python3' },
      language_info: { name: 'python', version: '3.10' },
    },
    nbformat: 4,
    nbformat_minor: 5,
  }
  writeFileSync(join(LAB, `day${day.id}`, `day${day.id}_전체.ipynb`), JSON.stringify(nb, null, 1), 'utf8')
  stats.nb = (stats.nb ?? 0) + 1
}

// ── 미니 프로젝트 완성 소스 ─────────────────────────────────
const pdir = join(LAB, 'projects')
rmSync(pdir, { recursive: true, force: true })
mkdirSync(pdir, { recursive: true })
for (const p of projects) {
  if (!p.solution) continue
  const header = [
    '# ' + '='.repeat(58),
    `# 미니 프로젝트 ${p.id} [${p.level}] — ${p.title}`,
    `# 환경: ${p.env}`,
    `# 목표: ${plain(p.goal)}`,
    `# 재사용: ${plain(p.reuse ?? '-')}`,
    '# 사이트: https://pytorch26.dreamitbiz.com/#/projects',
    '# ' + '='.repeat(58),
    ...(p.guide?.length ? ['# [진행 가이드]', ...p.guide.map((g) => `#  - ${plain(g)}`)] : []),
    ...(p.cert ? ['# [결과 인증]', `#  ${plain(p.cert)}`] : []),
    '',
    '',
  ].join('\n')
  writeFileSync(join(pdir, p.solution.filename), header + p.solution.source.trimEnd() + '\n', 'utf8')
  stats.proj++
}

// ── 설치 확인 스크립트 (사이트 환경설정 'check' 섹션 정본) ──
const check = setup.find((x) => x.id === 'check')
if (check?.code) {
  writeFileSync(join(LAB, '00_env_check.py'), check.code.source.trimEnd() + '\n', 'utf8')
}

console.log(JSON.stringify({ ...stats, days: days.length, course: course.title ?? null }, null, 2))
