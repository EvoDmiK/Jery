# 독립 평가셋(`frozen-eval-v1`) 누수·출처·정답 검토 감사기 및 공통 계약 피드백 보고서

## 1. 목적 및 배경

사용자 지시(Claude + Antigravity 공동 작업으로 고정 10개 도구당 10건씩 약 100건의 **신규 실제 사용자 요청** 평가셋 구축)와 루트 작업 계약(`outputs/frozen-eval-20261020/coordination-contract.md`, 기준 통합 커밋 `d40b844`)에 따라, `dev-agy` 담당 범위(`apps/review/`, `tests/review/`, `docs/review/`)에서 정제 후보 패킷(`cases.jsonl`), 매니페스트(`manifest.json`), 독립 AI 라벨 제안서(`proposals.json`)를 읽기 전용으로 검증하는 결정론적 감사기(`apps/review/frozen_eval_audit.py`)를 구현했다.

- **원칙 준수**: 원본 Hermes SQLite DB·설정·인증정보를 직접 열지 않으며, 합성 데이터를 실제 요청으로 위장(`fake100`)하거나 기존 `test.jsonl` 회귀 세트를 신규 독립 평가셋으로 재사용하지 않는다. 모델 예측을 참조해 정답을 정하지 않으며, AI 독립 제안은 `needs_human_review` 상태의 별도 파일로만 취급하고 사람 검토(`human_reviewed`)나 실행 검증(`execution_verified=true`)으로 승격하지 않는다.

---

## 2. 구현된 독립 감사기(`apps/review/frozen_eval_audit.py`) 검증 게이트

### 2.1 과거 학습·검증·테스트·보강 데이터 전체 대비 누수(Leakage) 차단
감사기는 다음 8개 기존 데이터 파일(총 **1,484건**, 코퍼스 지문 `5f7f245963f129f15142d333fa6ad5611b3a5c3218898aa74ed61bb28c28afc4`) 전체를 로드해 대조한다.

| 대상 파일 | 레코드 수 | SHA-256 |
|---|---:|---|
| `datasets/hermes-routing-v1/train.jsonl` | 78 | `99b93c5c223a69f2ee0ab32e2ab35db99721a36d41b4800cc21921ab1df36019` |
| `datasets/hermes-routing-v1/validation.jsonl` | 26 | `fb0e25670450c28e7ac177b64bac8be038d8b00135d1caeadab03d927c6b7f38` |
| `datasets/hermes-routing-v1/test.jsonl` | 26 | `a8ac5f27cbd29a676e8d576d285da9d5cdd4ac1d180ab38a922a70df2328781a` |
| `datasets/mcp-routing-v2/train.jsonl` | 668 | `5eb09bc5d05668f82fe669c36b44ec2bd4d758e46cdec9a4b8c73edd5ddb9bd9` |
| `datasets/mcp-routing-v2/validation.jsonl` | 246 | `6840efb974df1ddded18b6b27b765c1cfa43d182238eca2b6fae5c0c63d5ddf6` |
| `datasets/mcp-routing-v2/test.jsonl` | 248 | `7a1fc5600a24b265f6864914a567e03ba0bd18f5b53f896f692465c09df832c4` |
| `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl` | 180 | `a9650b720f1faa7f57fb61343e69a88222486e1bffb35dc6f0878c5b5c86954f` |
| `datasets/diagnostics/laya-ko-en-request-v1.json` | 12 | `7422e347b8d1f449b0ecad7326a65ef8b91616c1d4ca0660287f89b589eb2b75` |

- **차단 항목(`BLOCKER`)**:
  1. 패킷 내부 `id` 중복 및 정규화(`NFKC` + 소문자화 + 영숫자/한글 추출) 요청·상태 중복(`DUPLICATE_PACKET_IDS`, `DUPLICATE_PACKET_REQUESTS`)
  2. 기존 코퍼스와의 `id` 충돌(`HISTORICAL_ID_LEAKAGE`) 및 `scenario_id`/`session_group` 충돌(`HISTORICAL_SCENARIO_LEAKAGE`)
  3. 기존 코퍼스와의 정규화 `request` 또는 `(request, context)` 일치(`HISTORICAL_EXACT_REQUEST_LEAKAGE`)
- **근접 중복 경고(`WARNING`)**:
  - 단어 + 음절 3-gram Jaccard 유사도 `>= 0.55`(`--near-dup-threshold` 설정 가능)인 근접 변형 문장을 `HISTORICAL_NEAR_DUPLICATE_REQUESTS`로 검출해 변형 재사용을 방지한다.

### 2.2 출처 진위(`real` vs `synthetic`) 및 발화 주체(`human_user` vs `cron/worker/system`) 검증
- `source` 필드는 문자열이 아닌 객체(`{"kind": "real"|"synthetic", "actor": "human_user"|..., "provenance": "<가명 세션/로그 해시>"}`)여야 한다.
- `source.kind == "synthetic"`이 1건이라도 포함되면 `SYNTHETIC_SOURCE_IN_REAL_BENCHMARK`로 즉시 차단한다.
- `source.actor`가 `cron`, `worker`, `system`, `agent`, `automated`이거나 `request` 본문에 멀티에이전트 디스패치·스케줄러 프리앰블(`orca orchestration`, `worker_done`, `You are a dispatched worker` 등)이 감지되면 `NON_HUMAN_ACTOR_SOURCE`로 차단한다.

### 2.3 카탈로그 기능 정합성, 호출 전 맥락 경계, 개인정보(PII) 위생 검사
- **고정 10개 도구 카탈로그(`hermes-ten-tools-2289142ab57a`, SHA-256 `11c0253a5de892afaec5836b847cfb70bc76c4caf23e96b8b41a414183123847`)**:
  - 모든 사례의 `candidate_tools`가 10개 고정 도구와 정확히 일치하는지 확인한다.
  - `acceptable_choices`의 `tool`이 카탈로그 및 `candidate_tools` 내에 있는지, `action`이 카탈로그 `input_schema`의 `action.enum`을 준수하는지 검증한다(예: Obsidian 5개 도구는 필수 `action` enum 검증, Swallow 5개 도구는 `action` 속성이 없으므로 `action` 지정 시 거부).
- **호출 후 결과 및 정답 단서 누수 차단(`FEATURE_BOUNDARY_OR_POST_TOOL_LEAKAGE`)**:
  - `request`와 `context` 내부에 도구 ID(`mcp__*`)가 노출되거나 `context` 메시지에 `tool_call_id`, `tool_calls`, `outcome`, `observed_choice` 등 호출 후/메타데이터 키가 섞이면 즉시 차단한다.
- **미정제 개인정보·경로·시크릿 탐지(`UNSANITIZED_PII_OR_SECRET_DETECTED`)**:
  - 개인키 블록, Bearer/API 토큰, 자격증명 대입문, 마스킹되지 않은 이메일 주소, 로컬 절대 경로(`/Users/...`, `/Volumes/...` 등)가 남아 있으면 차단한다.

### 2.4 파일 동결 상태(`packet_status`)와 정답 검토 상태(`label_review_gate`)의 엄격한 분리
감사 결과 JSON은 파일 동결 여부와 라벨 검토 상태를 별도 필드로 보고한다.
- `packet_status`: `candidate` | `frozen` | `missing_prerequisites` (`freeze_implies_human_verified: false` 고정)
- `label_review_gate`:
  - `pending_initial_labels_needs_human_review` (`decision=null` 미해결 정답 존재)
  - `unresolved_gold_with_ai_proposals_needs_human_review` (`decision=null`이나 독립 AI 제안서가 첨부된 상태)
  - `needs_human_review` (라벨이 채워졌으나 `review_status != "human_reviewed"`이거나 100건 쿼터 미달)
  - `human_reviewed_complete` (10개 도구 × 10건 = 100건 모두 실제 인간 사용자 요청이며 `human_reviewed` 완료, 블로커 0건)
- `evaluation_gate`:
  - `refused`: 블로커가 존재하거나, 미해결(`null`) 정답이 있거나, `packet_status != "frozen"`인 경우
  - `provisional_exploratory_only_not_human_reviewed`: 패킷이 동결(`frozen`)되고 블로커가 없으나 사람 검토(`human_reviewed`)가 완료되지 않은 경우 (탐색적·잠정 보고만 허용, 검토된 실제 정확도 주장 금지)
  - `ready_for_reviewed_benchmark`: 패킷 동결 + 블로커 0건 + 10개 도구 × 10건 인간 검토 완료

---

## 3. 공통 계약(`coordination-contract.md`) 인터페이스 피드백 및 루트 전달 사항

루트(`dev-codex`)가 실제 Hermes DB에서 정제 후보 패킷을 추출·내보낼 때 다음 스키마 세부사항을 반영하면 감사기와 즉시 무결성 연동된다.

1. **`source` 객체 내 인간 사용자 vs 자동화 주체 구분 필드 표준화**:
   - `source: {"kind": "real", "actor": "human_user", "provenance": "session_<hmac>"}` 형식을 권장한다.
   - Hermes 세션 중 스케줄러/크론, 에이전트 워커 디스패치, 시스템 자동 트리거 세션은 수집 단계에서 제외하거나 `actor: "cron" | "worker" | "system"`으로 명시해 벤치마크 쿼터(100건)에 섞이지 않게 해야 한다.
2. **초기 후보 패킷(`status="candidate"`)의 미해결 정답 표기**:
   - 초기 패킷은 `decision: null`, `acceptable_choices: []`, `review_status: "needs_human_review"`, `execution_verified: false`로 시작하고, 실제 로그에서 관측된 호출은 `observed_choice`(단일 객체) 또는 `observed_choices`(배열)에만 보존한다.
3. **독립 AI 라벨 제안서(`proposals.json`) 구조**:
   - `cases.jsonl`을 직접 수정하지 않고 별도 `proposals.json`에 `{"review_status": "needs_human_review", "execution_verified": false, "proposals": {"<case_id>": [{"proposer": "claude"|"antigravity", "confidence": "high"|"medium"|"low", "disposition": "select_tools"|"ambiguous_multi_tool"|"exclude"|..., "rationale": "...", "decision": "select_tools", "acceptable_choices": [...]}]}}` 형태로 기록하면 감사기가 proposer별 분포와 만장일치 여부를 자동 집계한다.
4. **현재 루트 경로(`outputs/frozen-eval-20261020/`) 점검 결과**:
   - 현재 `/Volumes/Dove-Nest-SSD/projects/Jery/outputs/frozen-eval-20261020/`에는 `coordination-contract.md`만 존재하며, 정제된 `cases.jsonl`과 `manifest.json`은 루트의 Hermes 원본 인벤토리 및 정제 추출 대기 중이다.
   - 감사기를 해당 경로에 실행하면 가짜 데이터를 생성하지 않고 정확히 누락된 선행 조건(`Missing sanitized candidate cases file`, `Missing frozen-eval manifest file`)과 10개 도구별 부족분(총 100건 부족)을 보고하며 `evaluation_gate: "refused"`를 반환한다.

---

## 4. 재현 및 테스트 명령

```sh
# 1. 합성 픽스처 기반 감사기 단위 테스트 (7개 시나리오) 및 전체 리뷰 테스트 스위트 (총 53개 테스트)
python3 -m unittest discover -s tests/review -v

# 2. 루트 작업 디렉터리 대상 감사 실행 (현재 정제 패킷 대기 상태 확인)
python3 apps/review/frozen_eval_audit.py \
  --packet-dir /Volumes/Dove-Nest-SSD/projects/Jery/outputs/frozen-eval-20261020 \
  --json

# 3. 기본 데이터셋 무결성 검증
python3 scripts/validate_dataset.py
python3 scripts/validate_mcp_dataset.py
```
