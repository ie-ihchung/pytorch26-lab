/**
 * 주석을 보강한 소스를 사이트 dayN.js 의 code.source 에 되돌려 넣는다.
 *
 * 배경(2026-07-28 대표 지시):
 *   "내가 다 설명을 할 수 없으니 초등학생 이해 수준으로 낱줄에 주석" —
 *   강의 중 구두 설명이 닿지 않는 부분을 주석이 대신해야 한다.
 *
 * 입력: <SRC_DIR>/d<day>_<slot>.new.py 가 있으면 그 교시를 교체한다.
 * 교체 후에는 반드시 refresh_standalone.mjs 를 돌려 전체본도 갱신해야 한다
 * (전체본은 교시 코드를 이어 붙인 사본이라 원본만 고치면 어긋난다).
 *
 *   node scripts/apply_commented.mjs <SRC_DIR>
 *   node scripts/apply_commented.mjs <SRC_DIR> --dry
 */
import fs from 'node:fs'
import path from 'node:path'

const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'
const SRC_DIR = process.argv[2]
const DRY = process.argv.includes('--dry')

if (!SRC_DIR) { console.error('사용법: node scripts/apply_commented.mjs <SRC_DIR>'); process.exit(1) }

const BT = String.fromCharCode(96)
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

let changed = 0
for (const day of [1, 2, 3]) {
  const file = path.join(SITE, `src/data/day${day}.js`)
  if (!fs.existsSync(file)) continue
  let src = fs.readFileSync(file, 'utf8')
  let touched = false

  for (let slot = 1; slot <= 7; slot++) {
    const newFile = path.join(SRC_DIR, `d${day}_${slot}.new.py`)
    if (!fs.existsSync(newFile)) continue
    const body = fs.readFileSync(newFile, 'utf8').trimEnd()

    // 이 교시의 code: { ... source: `...` } 블록을 찾는다.
    // extraCode 의 source 와 헷갈리지 않도록 slot 위치부터 첫 filename 만 본다.
    const slotIdx = src.indexOf(`slot: ${slot},`)
    if (slotIdx < 0) { console.log(`  d${day}_${slot}: slot 못 찾음`); continue }

    const codeIdx = src.indexOf('      code: {', slotIdx)
    if (codeIdx < 0) { console.log(`  d${day}_${slot}: code 블록 없음`); continue }

    const srcKey = src.indexOf('        source: ' + BT, codeIdx)
    if (srcKey < 0) { console.log(`  d${day}_${slot}: source 없음`); continue }
    const start = srcKey + ('        source: ' + BT).length

    // 이스케이프되지 않은 닫는 백틱을 찾는다
    let end = start
    while (end < src.length) {
      if (src[end] === BT && src[end - 1] !== '\\') break
      end++
    }
    if (end >= src.length) { console.log(`  d${day}_${slot}: 닫는 백틱 못 찾음`); continue }

    const before = src.slice(start, end)
    const after = esc(body)
    if (before === after) { console.log(`  d${day}_${slot}: 동일, 건너뜀`); continue }

    src = src.slice(0, start) + after + src.slice(end)
    touched = true
    changed++
    const cnt = (s) => s.split('\n').filter((l) => l.trim() && !l.trim().startsWith('#')).length
    const cm = (s) => s.split('\n').filter((l) => l.includes('#')).length
    console.log(`  d${day}_${slot}  코드 ${cnt(body)}줄 / 주석 있는 줄 ${cm(body)}`)
  }

  if (touched && !DRY) fs.writeFileSync(file, src, 'utf8')
}

console.log(`\n${DRY ? '(미리보기) ' : ''}${changed}개 교시 교체`)
if (changed && !DRY) console.log('※ 이어서 node scripts/refresh_standalone.mjs 를 돌리세요')
