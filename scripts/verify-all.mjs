// 강의 전 전수 검증 — 사이트 데이터(정본)의 정합성만 검사한다.
// 파이썬 실행 검증은 scripts/verify-run.py 가 맡는다.
const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'

const { days, course } = await import(SITE + '/src/data/curriculum.js')
const projects = (await import(SITE + '/src/data/projects.js')).default
const quizzes = (await import(SITE + '/src/data/quizzes.js')).default

let fail = 0
const ok = (label, cond, detail = '') => {
  console.log(`${cond ? '[OK]  ' : '[실패]'} ${label}${detail ? ' — ' + detail : ''}`)
  if (!cond) fail++
}

console.log('── 커리큘럼 구조 ──')
ok('일차 3개', days.length === 3, `${days.length}일`)
for (const d of days) {
  ok(`${d.id}일차 교시 7개`, d.sessions.length === 7, `${d.sessions.length}교시`)
  const slots = d.sessions.map((s) => s.slot)
  ok(`${d.id}일차 교시 번호 1~7 중복 없음`, new Set(slots).size === 7 && Math.min(...slots) === 1 && Math.max(...slots) === 7)
  const codes = d.sessions.filter((s) => s.code)
  ok(`${d.id}일차 코드 블록`, codes.length === 7, `${codes.length}개`)
  ok(`${d.id}일차 코드 파일명 중복 없음`, new Set(codes.map((s) => s.code.filename)).size === codes.length)
}

console.log('\n── 미니 프로젝트 ──')
ok('프로젝트 8개', projects.length === 8, `${projects.length}개`)
ok('프로젝트 id 1~8 중복 없음', new Set(projects.map((p) => p.id)).size === 8)
ok('전 프로젝트 완성소스 보유', projects.every((p) => p.solution?.source))
ok('솔루션 파일명 중복 없음', new Set(projects.map((p) => p.solution.filename)).size === projects.length)

let blankBad = 0, blankTotal = 0
for (const p of projects) {
  for (const b of p.blanks ?? []) {
    blankTotal++
    const n = p.solution.source.split(b.find).length - 1
    if (n === 0 || (b.nth ?? 1) > n) blankBad++
  }
}
ok('괄호채우기 빈칸이 완성소스와 일치', blankBad === 0, `${blankTotal}개 중 불일치 ${blankBad}`)
ok('빈칸 난이도 1~3 범위', (projects.flatMap((p) => p.blanks ?? [])).every((b) => b.level >= 1 && b.level <= 3))

console.log('\n── 퀴즈 ──')
const qs = Array.isArray(quizzes) ? quizzes : Object.values(quizzes).flat()
ok('퀴즈 문항 존재', qs.length > 0, `${qs.length}문항`)
const badAnswer = qs.filter((q) => {
  const opts = q.options ?? q.choices
  if (!Array.isArray(opts)) return false
  const a = q.answer ?? q.correct
  return typeof a === 'number' && (a < 0 || a >= opts.length)
})
ok('정답 인덱스가 보기 범위 안', badAnswer.length === 0, `범위 밖 ${badAnswer.length}`)

console.log('\n── 코드 내용 점검 ──')
const allCode = [
  ...days.flatMap((d) => d.sessions.filter((s) => s.code).map((s) => ({ where: `${d.id}일차 ${s.slot}교시`, src: s.code.source }))),
  ...projects.map((p) => ({ where: `프로젝트 ${p.id}`, src: p.solution.source })),
]
const DEPRECATED = ['CliffWalking-v0', 'Taxi-v3', 'LunarLander-v2', 'Pendulum-v0', 'CartPole-v0']
for (const bad of DEPRECATED) {
  const hits = allCode.filter((c) => c.src.includes(bad))
  ok(`폐기 환경 ${bad} 미사용`, hits.length === 0, hits.map((h) => h.where).join(', '))
}
// matplotlib 한글 라벨을 쓰면서 폰트 지정이 없으면 강의 화면에서 글자가 깨진다
const hangul = /[가-힣]/
const plotKo = allCode.filter((c) => /plt\.(title|xlabel|ylabel|legend|suptitle)\([^)]*[가-힣]/.test(c.src))
const plotKoNoFont = plotKo.filter((c) => !c.src.includes('font.family'))
ok('한글 그래프 라벨에 폰트 지정', plotKoNoFont.length === 0, plotKoNoFont.map((c) => c.where).join(', ') || `한글라벨 ${plotKo.length}곳 전부 지정됨`)
// 종료 조건 없는 while True 는 무한 루프 사고의 원인이 된다
const infinite = allCode.filter((c) => /while True:/.test(c.src) && !/break|return/.test(c.src))
ok('while True에 탈출 경로 존재', infinite.length === 0, infinite.map((c) => c.where).join(', '))

console.log('\n── 사이트 메타 ──')
ok('과정명 존재', !!course.title, course.title)

console.log(`\n결과: ${fail === 0 ? '전 항목 통과' : `실패 ${fail}건`}`)
process.exit(fail ? 1 : 0)
