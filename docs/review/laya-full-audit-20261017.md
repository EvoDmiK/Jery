# Jery Laya 전체 후보 범위 익스포트 독립 감사 보고서 (Laya Full Export Audit - Revised)

- **감사 일시**: 2026-10-17
- **감사 주체**: `dev-agy` (데이터 검토 도구 및 결과 시각화 담당)
- **감사 대상**: `outputs/laya-full-20261017/export/`
  - `manifest.json` (크기: 2,999 bytes, SHA256: `ded2f8e2d2ee0035d164b04719c5c2dc0cda5e2bb815b0db355d024563803b7e`)
  - `train.jsonl` (4,172행 / 30,144,435 bytes, SHA256: `42a287bd32469f3aba284dfd6ea43ff373471caa253174cc9a9be36cbe488fad`)
  - `validation.jsonl` (1,592행 / 11,541,451 bytes, SHA256: `aa2c7175a6fb1564af7e992faa8aa64dce45418a9f9e32308a17e075815baf1b`)
  - `test.jsonl` (1,577행 / 11,434,645 bytes, SHA256: `1e13e7a57bb25dc80d835fea3af7ab3a1f01eb64a49c604cea354f766f8bebba`)
- **기반 문서 및 계약**:
  - `docs/contracts/decision-export-v1.md` (결정 모델 파생 학습 형식 계약 v1)
  - `datasets/mcp-routing-v2/` (원본 v2 데이터셋: train 668행, validation 248행, test 246행, 총 1,162행)
  - `docs/data/decision-training-strategy.md` (공동 통합 전략 문서)
- **작업 원칙**: 독립 감사 도구([`apps/review/full_audit.py`](../../apps/review/full_audit.py)) 및 격리된 임시 픽스처 기반 단위 테스트([`tests/review/test_full_audit.py`](../../tests/review/test_full_audit.py))를 자체 구현하여 검증 수행. 파이썬 표준 라이브러리(pure stdlib)만 사용하며 root 프로젝트에 대한 직접 수정, 학습 실행, 외부 패키지 설치 및 Git 커밋/푸시 일체 배제.

---

## 1. 감사 요약 및 종합 판정 (Executive Verdict)

### 종합 판정: PASS (감사 코드 검사 기준 전 항목 충족, 에러 0건)

Codex가 생성한 Laya 전체 후보 범위 학습용 익스포트(`decision-export-v1`, `purpose=full_synthetic_exploratory`, `export_scope=exhaustive_original_candidates`)에 대해 원본 데이터셋 1,162행 대조 및 파생된 7,341개 윈도우 행을 대상으로 정적 무결성 및 구조 정합성을 독립 감사한 결과:

1. **원본 분할 보존 및 3-Way 시나리오 격리**: 원본 train 668, validation 248, test 246행이 전수 보존되었으며, 3개 분할 간 시나리오 그룹 교차 누출(scenario leakage)이 0건임을 확인.
2. **원 후보 $C$ 전수 분할 및 구조적 커버리지**: 모든 원본 행의 후보 도구 목록 $C$가 최대 8개 단위의 상호 배타적 창 $W_i$로 분할되었으며, 누락·중복·허위 도구 없이 합집합 일치($\bigcup W_i = C$) 및 크기 합 일치($\sum |W_i| = |C|$)로 커버리지 1.0(100%)을 정적으로 충족함.
3. **원본 11개 필드 동일성 전수 일치**: 7,341행 전수에서 원본 요청 텍스트, 대화 맥락, 원 후보 도구 목록, 허용 정답, 시나리오 ID, 원본 결정, 카탈로그 버전, 검토 상태 및 출처 플래그가 원본과 100% 일치함을 확인.
4. **윈도우 의미론 및 타깃 확률 분포 정합성**: 양성 선택 창은 창 내 허용 도구 에일리어스에 균등 확률($1/K$)을 부여하고, 정답 도구가 포함되지 않은 자연 음성 창 및 원본 미지원(unsupported) 창은 정확히 `window_none` 1.0으로 라벨링됨.
5. **카드 필드 엄격 일치 및 코드 수준 텍스트 비누출 스캔**: 카드 필드가 정확히 4개(`alias, server, name, summary`)로 제한되고 요약문이 공백 정규화된 카탈로그 설명의 120자 접두사와 일치하며, `request`/`context` 내 96개 카탈로그 도구 ID 단순 부분문자열 및 5대 결정 키워드 정규식 토큰 검출 0건을 확인.
6. **출력 경로 보호 가드 통합**: `--output-json` 대상 경로가 데이터셋 디렉터리, 익스포트 디렉터리 또는 이들의 심볼릭/하드링크를 가리킬 경우 쓰기를 사전에 거부하는 검증 가드를 적용하여 원본 데이터 훼손을 방지.

```
[Full MCP-Routing-v2 Laya Export Architecture]
 ├── train (668 원본 사례 -> 4,172 윈도우 / 165 시나리오 그룹)
 │    ├── 1 창 사례:  344건 (344 창)
 │    ├── 2 창 사례:    6건 ( 12 창)
 │    └── 12 창 사례: 318건 (3,816 창)
 │    [타깃 라벨 분포] window_none: 3,409 | T0~T7: 607 | clarify: 78 | no_tool: 78
 ├── validation (248 원본 사례 -> 1,592 윈도우 / 88 시나리오 그룹)
 │    ├── 1 창 사례:  124건 (124 창)
 │    ├── 2 창 사례:    2건 (  4 창)
 │    └── 12 창 사례: 122건 (1,464 창)
 │    [타깃 라벨 분포] window_none: 1,305 | T0~T7: 235 | clarify: 26 | no_tool: 26
 └── test (246 원본 사례 -> 1,577 윈도우 / 87 시나리오 그룹)
      ├── 1 창 사례:  125건 (125 창)
      └── 12 창 사례: 121건 (1,452 창)
      [타깃 라벨 분포] window_none: 1,293 | T0~T7: 232 | clarify: 26 | no_tool: 26
 ──────────────────────────────────────────────────────────────────────────
  합계: 1,162 원본 사례 -> 7,341 파생 윈도우 / 340 시나리오 그룹 (에러 0건)
```

---

## 2. 독립 감사 도구 및 단위 테스트 구현 (Audit Tooling & Verification)

### 2.1 감사 도구 구현: [`apps/review/full_audit.py`](../../apps/review/full_audit.py)
- **표준 라이브러리 및 상대 경로 원칙 준수**: Python 3.9+ pure standard library만을 사용하여 외부 의존성 없이 동작하며, 기본 경로는 저장소 루트 상대경로(`DEFAULT_ROOT = Path(__file__).resolve().parents[2]`)로 지정하여 절대 Mac 경로 하드코딩을 제거함. 실감사는 CLI 인자(`--export-dir`, `--dataset-dir`)를 명시하여 실행.
- **출력 경로 덮어쓰기 보호 가드 (`validate_output_summary_path`)**:
  - `apps.review.audit`의 `validate_output_summary_path`를 재사용하여 `--output-json` 경로가 `dataset_dir` 또는 `export_dir` 내부이거나, 디렉터리 내 기존 파일(또는 그 심볼릭/하드링크)을 가리킬 경우 `ValueError`를 발생시키고 CLI 인자 파싱 단계에서 즉시 중단(`parser.error`).
- **카드 스키마 및 요약문 엄격 검사**:
  - 카드 객체의 키가 정확히 `{"alias", "server", "name", "summary"}` 4개와 일치하는지 검사 (`set(card.keys()) == CARD_WHITELIST_FIELDS`). 필드 누락(예: summary 부재)이나 비인가 필드 주입을 모두 차단.
  - `summary`가 카탈로그 설명의 공백 정규화 접두사(`" ".join(spec["description"].split())[:120]`)와 정확히 일치하는지 검사하여 빈 요약문(`""`)이나 미정규화 문자열의 통과를 방지.
- **코드 검사 범위 한정 (Anti-Leakage Scope)**:
  - 텍스트 검사는 카탈로그의 96개 전체 도구 ID(`mcp__<server>__<name>`)의 정확한 부분문자열 포함 여부와 5대 결정 키워드(`\b(select_tools|no_tool|clarify|unsupported|window_none)\b`)의 단어 경계 토큰 출현 여부만을 코드로 능동 스캔함. 임의의 자연어 의미 누출을 전수 보증하는 것은 아님을 명시.

### 2.2 단위 테스트 스위트: [`tests/review/test_full_audit.py`](../../tests/review/test_full_audit.py)
- **독립 격리 픽스처**: `tempfile.TemporaryDirectory` 내에 3개 분할 및 다중 창 픽스처(10개 후보 도구를 2개 창으로 분할)를 동적으로 생성하여 clean checkout 환경에서도 외부 파일 없이 독립 실행 가능.
- **신규 full audit 테스트 (15건)**:
  1. `test_clean_temp_fixture_passes`: 완전한 정합성 픽스처 100% 통과 확인.
  2. `test_missing_source_hash_key_detected`: `manifest.source_sha256`에 `test.jsonl` 누락 시 즉시 탐지.
  3. `test_split_sha256_mismatch_detected`: 분할 파일 변조 시 해시 불일치 즉시 탐지.
  4. `test_scenario_leakage_between_train_and_test_detected`: 분할 간 시나리오 누출 시 3-way 격리 위반 즉시 탐지.
  5. `test_source_fidelity_tampering_detected`: 요청 텍스트 임의 변조 시 원본 대조 실패 즉시 탐지.
  6. `test_missing_partition_window_detected`: 다중 창 중 일부 창 누락 시 창 개수 및 후보 누락 즉시 탐지.
  7. `test_fabricated_candidate_tool_in_window_detected`: 원본 $C$에 없는 허위 도구 주입 시 즉시 탐지.
  8. `test_window_none_label_violation_detected`: 정답 도구가 없는 창에 비인가 라벨 할당 시 의미론 위반 즉시 탐지.
  9. `test_card_summary_length_limit_detected`: 카드 요약이 120자를 초과할 때 즉시 탐지.
  10. `test_card_missing_required_key_detected`: 카드 키 중 `summary` 등 필수 키 누락 시 즉시 탐지.
  11. `test_card_empty_or_mismatched_summary_detected`: 빈 요약문(`""`)이나 미정규화 요약문 주입 시 즉시 탐지.
  12. `test_tool_id_leakage_detected`: `request` 또는 `context`에 도구 ID가 누출되었을 때 정규식 스캔으로 즉시 탐지.
  13. `test_cli_invocation_on_temp_fixture`: CLI 실행 및 JSON 결과 파일 저장 정상 동작 확인.
  14. `test_cli_refuses_to_overwrite_dataset_or_export_files`: `--output-json`이 데이터셋/익스포트 파일 또는 심볼릭 링크를 가리킬 때 비정상 종료 코드 반환 및 원본 바이트 불변 검증.
  15. `test_live_root_full_export_if_available`: 로컬 워크스페이스 내에 익스포트 파일이 존재할 경우에만 실측 검증 수행 (부재 시 skipTest로 정상 스킵).

### 2.3 전체 테스트 스위트 실행 결과 (Test Execution Breakdown)

```sh
python3 -m unittest discover -s tests/review -v
# Ran 36 tests in 2.513s -> OK (skipped=1)
```

- **기존 데이터 감사 테스트 (`test_audit.py`)**: 9건 통과
- **기존 파일럿 익스포트 감사 테스트 (`test_pilot_audit.py`)**: 12건 통과
- **신규 전체 익스포트 감사 테스트 (`test_full_audit.py`)**: 15건 중 14건 통과, 1건 스킵 (격리 워크스페이스 내 live 파일 부재 시 깨끗하게 스킵)

---

## 3. 원본 대조 및 8대 핵심 검증 항목 상세 실측 결과 (Audit Findings)

### 3.1 원본 동결 체크섬 대조 (Frozen Source Checksums)
`manifest.json`의 `source_sha256`에 4대 필수 파일이 전수 포함되어 있으며, `datasets/mcp-routing-v2` 원본 파일과의 바이트 단위 SHA256 체크섬이 일치:
- `train.jsonl`: `5eb09bc5d05668f82fe669c36b44ec2bd4d758e46cdec9a4b8c73edd5ddb9bd9` (일치)
- `validation.jsonl`: `6840efb974df1ddded18b6b27b765c1cfa43d182238eca2b6fae5c0c63d5ddf6` (일치)
- `test.jsonl`: `7a1fc5600a24b265f6864914a567e03ba0bd18f5b53f896f692465c09df832c4` (일치)
- `tools.json`: `d8a97f14b63fe9321f65942c6947d0c7d2370542b7cc9cb22f0683a42ca9a48b` (일치)
- **판정**: **PASS**

### 3.2 익스포트 분할 파일 무결성 및 해시 일치 (Split Checksums)
생성된 각 분할 파일의 SHA256 해시가 `manifest.json`의 `splits.<split>.sha256`과 정확히 일치:
- `train.jsonl`: `42a287bd32469f3aba284dfd6ea43ff373471caa253174cc9a9be36cbe488fad` (일치)
- `validation.jsonl`: `aa2c7175a6fb1564af7e992faa8aa64dce45418a9f9e32308a17e075815baf1b` (일치)
- `test.jsonl`: `1e13e7a57bb25dc80d835fea3af7ab3a1f01eb64a49c604cea354f766f8bebba` (일치)
- **판정**: **PASS**

### 3.3 분할 격리 및 3-Way 시나리오 그룹 교차 검증 (Scenario Isolation)
- `train`: 고유 `scenario_id` 165개
- `validation`: 고유 `scenario_id` 88개
- `test`: 고유 `scenario_id` 87개
- **3-Way 교차 검사**: `train & val == set()`, `train & test == set()`, `val & test == set()`
  - 총 340개 시나리오 그룹이 상호 배타적으로 유지됨.
- **판정**: **PASS**

### 3.4 원본 사례 수 대 파생 윈도우 수 구별 (Source Cases vs. Windows)

| 분할 (Split) | 원본 사례 수 (Source Cases) | 창 수 (Window Rows) | 시나리오 그룹 수 | 창 분할 분포 (1창 / 2창 / 12창) |
|---|---|---|---|---|
| **train** | 668건 | 4,172행 | 165개 | 1창: 344건 / 2창: 6건 / 12창: 318건 |
| **validation** | 248건 | 1,592행 | 88개 | 1창: 124건 / 2창: 2건 / 12창: 122건 |
| **test** | 246건 | 1,577행 | 87개 | 1창: 125건 / 2창: 0건 / 12창: 121건 |
| **합계** | **1,162건** | **7,341행** | **340개** | **1창: 593건 / 2창: 8건 / 12창: 561건** |

- 96개 후보 도구를 가진 사례는 12개의 8도구 창으로 분할되었으며, 8개 이하 후보를 가진 사례는 1개 창(9~16개 도구는 2개 창)으로 분할됨을 확인.
- **판정**: **PASS**

### 3.5 원 후보 $C$ 전수 분할 및 구조적 커버리지 검증 (Candidate Partitioning & Coverage)
1. **창 크기 제약**: 7,341행 전수에서 $1 \le |W_i| \le 8$ 준수 확인.
2. **상호 배타적 분할 (Disjoint Partitioning)**: 각 원본 사례의 창 도구 수 합이 원본 후보 도구 수와 일치 ($\sum_{i=0}^{K-1} |W_i| = |C|$, 중복 도구 0건).
3. **완전한 합집합 (Exhaustive Union)**: $\bigcup_{i=0}^{K-1} W_i = \text{set}(C)$ (누락 도구 0건).
4. **허위 도구 부재 (No Hallucination)**: 원본 $C$에 없는 도구가 $W_i$에 포함된 사례 0건.
5. **창 인덱스 연속성**: 모든 사례에서 `window_index`가 $0$부터 `window_count - 1`까지 연속적으로 부여됨.
6. **후보 커버리지 지표**: train 1.0 (668/668), val 1.0 (248/248), test 1.0 (246/246) 달성.
- **주의 및 절차적 구분**:
  - 본 감사 스크립트가 검증한 것은 익스포트된 창들이 원본 후보 $C$를 누락/허위 없이 정적으로 전수 커버한다는 사실이다.
  - 창 분할이 정답 라벨을 보지 않고(gold-blind) 수행되었음은 원본 변환 코드(`build_laya_full_dataset.py`)의 순서(타깃 평가 전 셔플 및 분할)와 root의 별도 단위 테스트(`test_laya_full_export.py`의 `test_partitions_and_model_input_do_not_use_labels`)를 통해 입증되는 영역이다. 감사 스크립트의 정적 커버리지 결과가 절차적 블라인드 자체를 직접 증명하는 것은 아님을 명확히 기록한다.
- **판정**: **PASS**

### 3.6 원본 11개 필드 동일성 전수 대조 (Source Row Fidelity)
익스포트된 7,341행 전수에 대해 `metadata.source_id`를 기반으로 원본 v2 데이터셋 행을 직접 조회하여 11개 필드의 값 동일성을 검증:
1. `state.request == raw.request` (사용자 요청 텍스트 일치)
2. `state.context == raw.context` (대화 컨텍스트 일치)
3. `metadata.original_candidate_tools == raw.candidate_tools` (원본 $C$ 일치)
4. `metadata.acceptable_choices == raw.acceptable_choices` (허용 정답 일치)
5. `metadata.scenario_id == raw.scenario_id` (시나리오 식별자 일치)
6. `metadata.original_decision == raw.decision` (원본 결정 유형 일치)
7. `metadata.tool_catalog_version == raw.tool_catalog_version` (카탈로그 버전 일치)
8. `metadata.source == raw.source` (데이터 출처 일치)
9. `metadata.review_status == raw.review_status` (검토 상태 일치)
10. `metadata.execution_verified == raw.execution_verified` (실행 검증 여부 일치)
11. `metadata.source_split == raw.split` (원본 분할 배정 일치)
- **결과**: 7,341행 전 행에서 원본 데이터와의 불일치 **0건**.
- **판정**: **PASS**

### 3.7 윈도우 의미론, 타깃 라벨 및 소프트 타깃 확률 정규화 (Window Semantics & Targets)
1. **양성 선택 창 (Positive Select Windows, 1,074건)**:
   - train 607건, val 235건, test 232건.
   - 창 $W$ 안에 원본 `acceptable_choices` 도구가 포함된 경우, 정답 라벨이 해당 창의 에일리어스(`T0`~`T7`)로 지정됨.
   - 단일 허용 도구인 경우 확률 1.0 부여.
   - **복수 대안 창 (Multi-target Windows, 8건)**: train 5건, val 1건, test 2건에서 2개의 서로 다른 허용 도구가 동일 창에 배치되어 각 에일리어스에 0.5씩 균등 분배됨(`abs(sum(p) - 1.0) < 1e-6`).
2. **자연 음성 창 (Natural Negative Windows, 5,837건)**:
   - train 3,299건, val 1,275건, test 1,263건.
   - 원본 결정은 `select_tools`이지만 해당 창에는 정답 도구가 없는 경우, 정답 라벨은 **`window_none`** (확률 `{"window_none": 1.0}`).
   - 메타데이터의 `original_decision`은 여전히 `select_tools`로 보존되며, `derived_negative_window`는 파일럿의 수작업 제거 창과 구별하기 위해 `false`로 유지됨.
3. **원본 미지원 창 (Unsupported Windows, 170건)**:
   - train 110건(44사례), val 30건(8사례), test 30건(8사례). 모든 창의 정답 라벨이 **`window_none`** (확률 1.0).
4. **무도구(no_tool, 130건) 및 명확화(clarify, 130건)**:
   - train 각 78건(12사례), val 각 26건(4사례), test 각 26건(4사례). 모든 분할 창에서 정답 라벨이 각각 **`no_tool`**, **`clarify`** (확률 1.0).
- **판정**: **PASS**

### 3.8 카드 필드 엄격 일치 및 코드 수준 텍스트 비누출 스캔 (Sanitization & Anti-Leakage)
1. **카드 필드 엄격 일치**: `state.candidates` 내 모든 카드 객체는 정확히 `alias`, `server`, `name`, `summary`의 4개 필드만을 가짐 (누락 0건, 비인가 필드 0건).
2. **요약문 공백 정규화 접두사 및 120자 제한**: 7,341행 전수에서 `len(summary) <= 120`을 만족하며, 카탈로그(`tools.json`) 설명의 공백 정규화 접두사(`" ".join(desc.split())[:120]`)와 정확히 일치 (빈 요약문 0건).
3. **에일리어스 무결성**: 모든 창에서 카드 에일리어스가 `T0`부터 순차적으로 부여되었으며, 중복 에일리어스 0건.
4. **질문 허용목록**: `questions` 딕셔너리에 `route_in_window` 외의 비인가 질문 부재 확인.
5. **텍스트 비누출 스캔 (실제 코드 검사 범위)**:
   - `state.request` 및 `state.context` 텍스트를 대상으로 카탈로그의 전체 96개 도구 ID(`mcp__<server>__<name>`) 부분문자열 포함 여부와 5대 결정 키워드(`\b(select_tools|no_tool|clarify|unsupported|window_none)\b`) 토큰 출현 여부를 전수 스캔한 결과 **검출 0건**.
   - 주의: 본 검사는 정의된 토큰/문자열 패턴 검사이며, 임의의 자연어 맥락 누출 전체를 보장하는 것은 아님.
6. **미검토 플래그 불변성 보존**:
   - `source == 'synthetic'`: 7,341행 (100%)
   - `review_status == 'needs_human_review'`: 7,341행 (100%)
   - `execution_verified == False`: 7,341행 (100%)
- **판정**: **PASS**

---

## 4. 데이터셋 통계 및 결정 라벨 요약표

### 4.1 분할별 원본 결정 분포 (Source Decision Counts)

| 분할 (Split) | select_tools | clarify | unsupported | no_tool | 총 원본 사례 수 |
|---|---|---|---|---|---|
| **train** | 600 | 12 | 44 | 12 | 668 |
| **validation** | 232 | 4 | 8 | 4 | 248 |
| **test** | 230 | 4 | 8 | 4 | 246 |
| **합계** | **1,062** | **20** | **60** | **20** | **1,162** |

### 4.2 분할별 윈도우 정답 라벨 분포 (Window Target Labels)

| 라벨 (Label) | train | validation | test | 전체 창 수 |
|---|---|---|---|---|
| **window_none** | 3,409 (81.7%) | 1,305 (82.0%) | 1,293 (82.0%) | 6,007 (81.8%) |
| **T0** | 81 | 46 | 35 | 162 |
| **T1** | 71 | 27 | 29 | 127 |
| **T2** | 86 | 27 | 31 | 144 |
| **T3** | 74 | 26 | 31 | 131 |
| **T4** | 68 | 21 | 23 | 112 |
| **T5** | 75 | 30 | 33 | 138 |
| **T6** | 77 | 31 | 24 | 132 |
| **T7** | 75 | 27 | 26 | 128 |
| **clarify** | 78 | 26 | 26 | 130 |
| **no_tool** | 78 | 26 | 26 | 130 |
| **총 윈도우 행 수** | **4,172** | **1,592** | **1,577** | **7,341** |

---

## 5. 학습 시 주의사항 및 한계점 (Limitations & Downstream Guidance)

1. **토큰 예산 실측 필요 (Token Budget Verification)**:
   - 카드의 `summary`가 120자 제한을 충족하지만, Laya 모델의 실제 tokenizer 기준 토큰 수 제한을 통과하는지는 모델 담당자(Claude)가 본 학습 실행 전에 전수 측정해야 함. 향후 실제 budget에 맞춰 새 export가 수정/재생성되면 해당 규격에 맞춰 재감사를 수행할 예정임 (현재 감사 스코프: 120자 접두사 규칙).
2. **음성 창 불균형 (Negative Window Class Imbalance)**:
   - 96개 후보가 12개 창으로 분할되면서 자연 음성 창(`window_none`)이 전체 학습 데이터의 **81.7%**(3,409 / 4,172행)를 차지함. 모델이 상시 `window_none`만 예측하는 붕괴를 방지하기 위해 손실 가중치(class weighting) 또는 네거티브 창 샘플링 전략이 고려되어야 함.
3. **창 간 Softmax 점수 직접 비교 불가**:
   - 각 창은 독립적인 분류 컨텍스트에서 예측되므로 서로 다른 창에서 도출된 softmax 확률값의 절대 크기를 직접 비교할 수 없음. 원본 사례 단위의 최종 도구 선택을 위해서는 별도의 집계(aggregation) 또는 앙상블 규칙이 모델 평가 단계에서 정의되어야 함.
4. **테스트 분할의 성격**:
   - test 분할(246건)은 이전 베이스라인 실험에서도 관찰된 바 있으므로 회귀 평가(regression evaluation) 목적으로 활용되어야 하며, 완전히 새로운 외부 일반화 증거로 단정하지 않음.
5. **합성 미검토 데이터의 한계**:
   - 모든 사례는 `synthetic` 및 `needs_human_review` 상태를 유지하고 있으므로, 본 익스포트의 학습 및 평가 결과는 모델 아키텍처 및 공학적 파이프라인 검증용으로 해석되어야 하며 실제 운영 환경의 최종 정확도로 주장할 수 없음.

---

## 6. 결론 (Conclusion)

Codex가 생성한 `outputs/laya-full-20261017/export`는 계약(`decision-export-v1`)과 원본 데이터셋(`mcp-routing-v2`)의 명세를 정적으로 100% 충족함을 확인하였다. `dev-agy`는 감사 도구에 `--output-json` 덮어쓰기 방지 가드를 통합하고, 카드 필수 키 4개 및 정확한 공백 정규화 요약문 검증을 강화하였으며, 절대 경로 하드코딩을 제거하였다. 격리된 단위 테스트 15건(전체 36건 중 35건 통과, 1건 안전 스킵)을 확보하고 실측 요약 JSON을 갱신함으로써 전체 후보 범위 익스포트에 대한 독립 감사 임무를 완수하였다.
