# Jery Laya 파일럿 32건 공학용 익스포트 독립 감사 보고서 (Laya Pilot Audit - Revised)

- **감사 일시**: 2026-10-17
- **감사 주체**: `dev-agy` (데이터 검토 도구 및 결과 시각화 담당)
- **감사 대상**: `outputs/laya-pilot-20261017/export/`
  - `manifest.json` (크기: 1,999 bytes, SHA256: `2dab85790499d3e2006eab0a534d54c97bad41725d1e86faa8bb40c723a4674b`)
  - `train.jsonl` (32행 / 108,940 bytes, SHA256: `10769b08630c102ed118343a09b6b3b78eadc47929ca673421fd8ccdd18369ab`)
  - `validation.jsonl` (12행 / 41,056 bytes, SHA256: `a2bfd802343e0d207e6fa5da5efb4d3508d626b1490da1b124a60a2a1f204e0e`)
- **기반 문서 및 계약**:
  - `docs/data/decision-training-strategy.md` (공동 통합 전략 문서)
  - `docs/contracts/prediction-v1.md` (공통 예측 계약 v1)
  - `datasets/mcp-routing-v2/` (원본 v2 데이터셋)
- **작업 원칙**: 독립 감사 도구(`apps/review/pilot_audit.py`) 및 격리된 임시 픽스처 기반 단위 테스트(`tests/review/test_pilot_audit.py`)를 자체 구현하여 검증 수행. 코드 학습·패키지 다운로드·Git 커밋/푸시 일체 배제.

---

## 1. 감사 요약 및 종합 판정 (Executive Verdict)

### 종합 판정: PASS (전 항목 검증 기준 충족, 에러 0건)

Codex가 생성한 Laya 파일럿 32건 학습용 및 12건 검증용 익스포트(`decision-export-v1`)에 대해 원본 데이터셋 대조 및 구조 무결성을 전수 감사한 결과, **원본 필드 동일성, 윈도우 의미론, 시나리오 그룹 분리, 카드 필드 허용목록 준수, 데이터 비누출 및 출처 추적성 전 부문에서 규격을 충족**함을 확인하였다.

```
[Laya Pilot 32-Case Export Structure]
 ├── train.jsonl (32행 / 26 시나리오 그룹)
 │    ├── select_tools (양성 창): 12행 (내부 라벨: T0~T7)
 │    ├── select_tools (음성 창): 4행  (내부 라벨: window_none)
 │    ├── no_tool:               6행  (내부 라벨: no_tool)
 │    ├── clarify:               5행  (내부 라벨: clarify)
 │    └── unsupported:           5행  (내부 라벨: window_none)
 └── validation.jsonl (12행 / 10 시나리오 그룹)
      ├── select_tools (양성 창): 4행  (내부 라벨: T0~T6)
      ├── select_tools (음성 창): 2행  (내부 라벨: window_none)
      ├── no_tool:               2행  (내부 라벨: no_tool)
      ├── clarify:               2행  (내부 라벨: clarify)
      └── unsupported:           2행  (내부 라벨: window_none)
```

---

## 2. 독립 감사 도구 및 격리된 단위 테스트 (Audit Tooling & Clean Tests)

`dev-agy`는 파이썬 표준 라이브러리(Python 3.9+ pure stdlib)만을 사용하여 clean checkout 환경에서도 외부 파일이나 절대경로에 의존하지 않고 재현 가능한 감사 도구와 단위 테스트를 구현하였다:

1. **독립 감사 스크립트**: [`apps/review/pilot_audit.py`](../../apps/review/pilot_audit.py)
   - 상대경로(`DEFAULT_ROOT = Path(__file__).resolve().parents[2]`)를 기본값으로 사용하며, 특정 사용자 경로 하드코딩 없이 `--export-dir` 및 `--dataset-dir` CLI 인자를 명시적으로 지원.
   - 원본 체크섬 대조, 원본 11개 필드 동일성 대조, 분할 격리, $C$ vs $W$ 매핑, `window_none` 의미론, 동적 에일리어스, 소프트 타깃 정규화, `request`/`context`의 도구 ID·결정 키워드 검사, 카드 필드 허용목록 및 120자 제한 검증 수행.
2. **단위 테스트 스위트**: [`tests/review/test_pilot_audit.py`](../../tests/review/test_pilot_audit.py)
   - `tempfile.TemporaryDirectory` 내에 독립적인 미니 데이터셋과 익스포트 픽스처를 동적 생성하여 root outputs 유무와 무관하게 완전히 격리된 테스트 실행.
   - **결함 주입 및 탐지 테스트 9종 포함**:
     1. 체크섬 키 누락 탐지 (`tools.json` 누락 시 즉시 실패)
     2. 악성/변조된 `source_id` 탐지 (원본 미존재 ID 참조 시 실패)
     3. `state.context` 변조 탐지 (원본 context와 불일치 시 실패)
     4. `state.request` 변조 탐지 (원본 request와 불일치 시 실패)
     5. 카드 요약 120자 초과 탐지
     6. 카드 허용목록 외 비인가 필드 주입 탐지
     7. 중복 카드 에일리어스 탐지
     8. 비인가 질문(`action_choice` 등) 주입 탐지
     9. train-validation 간 시나리오 그룹 누출 탐지
   - 실제 환경의 root export가 존재하는 경우에만 선택적으로 실측 검증하는 가드형 테스트(`test_live_root_export_if_available`) 포함.
3. **전체 테스트 실행 결과**:
   ```sh
   python3 -m unittest discover -s tests/review -v
   # Ran 21 tests in 2.343s -> OK (기존 9건 + 신규 12건 전수 통과)
   ```

---

## 3. 원본 대조 및 8대 핵심 검증 항목 상세 결과 (Detailed Audit Findings)

### 3.1 원본 동결 체크섬 대조 (Frozen Source Checksums)
- `manifest.json`의 `source_sha256`에 필수 키(`train.jsonl`, `validation.jsonl`, `tools.json`)가 전수 포함되어 있는지 검증.
- 실제 `datasets/mcp-routing-v2` 파일의 바이트 단위 SHA256 대조:
  - `train.jsonl`: `5eb09bc5d05668f82fe669c36b44ec2bd4d758e46cdec9a4b8c73edd5ddb9bd9` (일치)
  - `validation.jsonl`: `6840efb974df1ddded18b6b27b765c1cfa43d182238eca2b6fae5c0c63d5ddf6` (일치)
  - `tools.json`: `d8a97f14b63fe9321f65942c6947d0c7d2370542b7cc9cb22f0683a42ca9a48b` (일치)
- **판정**: **PASS**

### 3.2 원본 데이터셋 11개 필드 동일성 전수 대조 (Source Row Fidelity)
익스포트된 44행(train 32 + val 12) 전부에 대해 `source_id`를 기반으로 원본 `datasets/mcp-routing-v2/{train,validation}.jsonl`의 해당 행을 직접 조회하여 아래 11개 필드의 완전 동일성(Equality)을 검증:
1. `state.request == raw.request` (사용자 질의 텍스트 100% 동일)
2. `state.context == raw.context` (이전 대화 맥락 리스트 100% 동일)
3. `metadata.original_candidate_tools == raw.candidate_tools` (원본 $C$ 목록 100% 동일)
4. `metadata.acceptable_choices == raw.acceptable_choices` (허용 정답 목록 100% 동일)
5. `metadata.scenario_id == raw.scenario_id` (시나리오 식별자 100% 동일)
6. `metadata.original_decision == raw.decision` (원본 결정 유형 100% 동일)
7. `metadata.tool_catalog_version == raw.tool_catalog_version` (카탈로그 버전 100% 동일)
8. `metadata.source == raw.source` (데이터 출처 100% 동일)
9. `metadata.review_status == raw.review_status` (검토 상태 100% 동일)
10. `metadata.execution_verified == raw.execution_verified` (실행 검증 여부 100% 동일)
11. `metadata.source_split == raw.split` (원본 분할 배정 100% 동일)
- **판정**: **PASS** (44행 전 행에서 원본 데이터와의 불일치 0건)

### 3.3 분할 구조 및 시나리오 그룹 격리 (Split Isolation)
- `train`: 32행, 고유 `scenario_id` 26개
- `validation`: 12행, 고유 `scenario_id` 10개
- **시나리오 누출 검사**: `set(train_scenarios) & set(val_scenarios) == set()`
  - 두 분할 간 시나리오 패밀리 교차 누출 0건 확인 (완전한 Group-disjoint 달성).
- **판정**: **PASS**

### 3.4 원본 $C$ 대 창 $W$ 매핑 및 `window_none` 의미론 (Window Semantics)
- 모든 사례에서 $1 \le |W| \le 8$ 준수 확인.
- 모든 사례에서 $W \subseteq C$ 준수 확인 (원본 후보 밖의 허위 도구가 창에 주입된 사례 0건).
- **의미론 검증**:
  1. **양성 선택 창 (Positive Select, 16건)**:
     - train 12건, validation 4건.
     - 원본 `acceptable_choices` 도구가 $W$에 포함되어 있으며, 정답 라벨이 해당 도구의 에일리어스(`T0`~`T7`)로 정확히 지정됨.
     - `window_none`은 정답 타깃에 포함되지 않음.
  2. **파생 음성 창 (Negative Windows, 6건)**:
     - train 4건, validation 2건.
     - 원본 `select_tools` 사례에서 정답 도구를 고의 제외한 창으로, $W$ 내부 도구 중 허용 도구가 0건임을 확인.
     - 정답 라벨: **`window_none`** (확률 `{"window_none": 1.0}`).
  3. **원본 미지원 사례 (unsupported, 7건)**:
     - train 5건, validation 2건.
     - $C$ 전체로 처리 불가한 사례이므로 창 $W$에서도 정답 도구가 없으며, 내부 라벨이 비표준 라벨이 아닌 **`window_none`**으로 지정됨 (확률 `{"window_none": 1.0}`).
  4. **무도구(no_tool, 8건) 및 명확화(clarify, 7건)**:
     - train no_tool 6건 / clarify 5건, validation no_tool 2건 / clarify 2건.
     - 정답 라벨이 각각 `no_tool`, `clarify`로 1.0 확률 단독 할당.
- **판정**: **PASS**

### 3.5 동적 에일리어스 및 다중 대안 소프트 타깃 (Aliases & Soft Targets)
- 모든 창 $W$의 도구들이 `T0`부터 `T{k-1}`까지 순차적으로 1:1 매핑됨 확인.
- `questions.route_in_window.criteria`의 선택지는 항상 해당 에일리어스 목록 + `no_tool`, `clarify`, `window_none`으로 구성됨.
- **소프트 타깃 확률 정규화**:
  - 모든 행의 `probabilities` 합이 정확히 $1.0$ ($\pm 1e-6$)을 유지.
  - 다중 대안 지원 로직: `build_laya_pilot.py`에서 `acceptable`을 `set`으로 도구 레벨에서 먼저 중복 제거(deduplicate)하여, 동일 도구의 여러 액션이 존재하더라도 특정 도구의 에일리어스에 가중치가 왜곡 집중되지 않도록 설계됨을 코드 레벨에서 확인.
- **판정**: **PASS**

### 3.6 상태 및 질문 비누출 검사 (Sanitization & No Data Leakage)
- `state` 필드 구조: 오직 `request`, `context`, `candidates` 3개 키만 허용.
- **누출 패턴 정규식 전수 검사**:
  - `state.request` 및 `state.context`(직렬화 문자열 포함) 전반을 카탈로그 96개 전체 도구 ID(`mcp__...`)와 대조하여 도구 ID 누출 0건 확인.
  - 결정 유형 키워드(`select_tools`, `no_tool`, `clarify`, `unsupported`, `window_none`)가 정답 유출 단어로 단독 등장하는지 검사하여 이상 없음 확인.
- `questions.route_in_window.criteria`의 설명문이 `도구 T0`, `도구 T1` 등의 중립적 템플릿으로 작성되어 정답 힌트를 제공하지 않음 확인.
- **판정**: **PASS**

### 3.7 도구 카드 필드 허용목록 및 120자 제한 검증 (Card Spec Compliance)
- `state.candidates` 내의 모든 카드에 대해:
  1. **필드 허용목록(Whitelist) 검사**: 각 카드는 오직 `alias`, `server`, `name`, `summary` 4개 필드만 포함하며, 비인가 필드가 주입되지 않음 확인.
  2. **에일리어스 고유성(Uniqueness) 검사**: 창 내에서 카드 에일리어스 중복이 존재하지 않으며, `alias_to_tool` 매핑과 1:1 일치.
  3. **요약문 120자 제한 검사**: 모든 카드의 `len(summary) <= 120` 준수 확인.
  4. **카탈로그 설명문 접두사 검사**: `summary`가 카탈로그 원본 `description`(공백 정규화)의 정확한 접두사(Prefix)임을 확인.
- **판정**: **PASS**

### 3.8 비인가 질문 배제 및 출처 메타데이터 보존 (No Extra Questions & Frozen Provenance)
- **질문 허용목록 검사**: `questions` 딕셔너리에는 오직 `route_in_window`만 존재하며, `action_choice` 등의 비인가 질문이 포함되지 않음 확인 (`action_prediction: false`).
- **출처 및 검증 플래그 보존**: 44행 전 행의 `metadata`에서 `source == 'synthetic'`, `review_status == 'needs_human_review'`, `execution_verified == False`가 100% 엄격히 유지됨 확인.
- **판정**: **PASS**

---

## 4. 감사 관찰 사실 및 파일럿 한계 (Audit Observations & Limitations)

### 4.1 후보 개수 지름길($K=4$) 잔존 (Confounder Remains)
- 원본 v2 데이터셋의 특성으로 인해, 이번 44건 파일럿 익스포트에서도 **비선택 17건(clarify 7건, unsupported 7건, no_tool 3건)의 원본 후보 개수 $C$가 4개**로 고정되어 있다.
  - train: $C=4$인 12건이 모두 non-select (clarify 5, unsupported 5, no_tool 2).
  - validation: $C=4$인 5건이 모두 non-select (clarify 2, unsupported 2, no_tool 1).
- 이는 v2 합성 데이터의 한계이며, `manifest.json`의 `limitations`에 `"Limited distinct nonselect scenario groups; candidate-count confounding remains"`로 정확히 명시되어 있다.
- 본 픽스처는 모델의 일반화 품질이나 캘리브레이션을 입증하기 위한 데이터가 아니며, **"파이프라인 입출력 변환, 토크나이저 예산 및 forward/backward 학습 엔지니어링 스모크 테스트"** 전용이므로 파일럿 착수를 가로막지 않는다.

### 4.2 실제 토크나이징 및 하드웨어 프로파일링의 주체 (Model Team Domain)
- 코디네이터의 사전 진단에 따르면, 공식 토크나이저 기준 컴팩트 JSON state 토큰 수는 train 기준 min 191 / median 397.5 / max 677 토큰, 옵션+마커 토큰 합산 최대 약 104 토큰으로 나타나 256 옵션 예산 및 1,024 전체 시퀀스 예산 내에 충분히 안착할 가능성이 확인되었다.
- 네이티브 인코더 레이아웃에서의 실제 텐서 입력 잘림 비율 및 M4 MPS/MLX 런타임 프로파일링은 `dev-claude`의 소관으로 최종 실측되어야 하며, `dev-agy`는 임의의 하드웨어 추정치를 단언하지 않는다.

---

## 5. 결론 및 다음 단계 (Conclusion & Next Steps)

1. **감사 최종 결론**:
   - Codex가 빌드한 `outputs/laya-pilot-20261017/export/`는 원본 `datasets/mcp-routing-v2`와 검사한 11개 원본 필드가 일치하며, 윈도우 의미론, 데이터 누출 방지, 카드 명세 준수, 시나리오 그룹 분리 등 모든 검증 기준을 충족한다.
   - 따라서 **Laya 공학용 파일럿 0단계(32건 스모크 테스트)를 실행 경로 확인을 진행하기에 충분**하다.

2. **다음 단계 권고사항**:
   - 실제 토큰 예산과 학습·저장·복원 결과는 [통합 파일럿 보고서](../data/laya-pilot-20261017.md)에 기록했다.
   - `dev-agy`는 파일럿 결과가 도출되면 오류 분석 뷰어 및 검토 UI 지원을 준비한다.

검사 범위는 이 합성 파일럿의 구조·계보·알려진 문자열 패턴이다. 원본 요청 텍스트의 모든 의미적 정답 누출 부재나 사람에 의한 정답 검토를 입증하지 않는다.

통합 검증에서는 기존 감사 도구의 출력 경로 보호를 재사용했다. 원본 데이터·export 파일과 그 심볼릭 링크·하드 링크를 결과 출력으로 지정해도 거부하고 원본 바이트를 보존하는 CLI 테스트를 추가했다. 파일럿 감사 테스트 13개가 통과했다.
