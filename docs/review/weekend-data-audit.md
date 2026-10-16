# Jery MCP 라우팅 데이터셋 v2 독립 감사 보고서 (Weekend Data Audit)

- **감사 일시**: 2026-10-16
- **감사 대상**: `datasets/mcp-routing-v2`
- **카탈로그 버전**: `hermes-mcp-57e701d87790` (등록 MCP 서버 5개, 도구 96개)
- **감사 주체**: `dev-agy` (데이터 검토 도구 및 시각화 담당)
- **감사 원칙**: 읽기 전용 독립 감사, Ponytail 최소 구현 원칙(Python stdlib만 사용), 데이터 원본 보존, 실행 검증과 검토 상태 독립 표기

---

## 1. 감사 요약 및 판정 (Executive Summary)

### 1.1 총평: 조건부 진행 (Conditional Proceed with Specific Guards)
`datasets/mcp-routing-v2` (1,162개 사례, 시나리오 340개 그룹)는 `select_tools` 결정 유형과 96개 도구 라우팅의 전반적인 구조적 정합성을 갖추고 있으나, **주말 기준 모델 학습 및 평가 시 반드시 사전에 격리하거나 보정해야 할 2개의 차단성 결함(Blocker)과 4개의 왜곡 위험(Warning)**이 식별되었다.

이러한 결함들은 데이터셋 원본 수정 없이도 **모델 학습 설정 및 평가 지표 집계 방식에서 보정 가능**하므로, 본 보고서의 가이드라인에 따라 평가 프로토콜을 설정한 후 탐색 모델 학습을 진행할 것을 권고한다.

### 1.2 데이터 상태 및 검증 플래그 (독립 표기 원칙 준수)
- `source`: **전체 `synthetic`** (1,162/1,162건, 100%)
- `review_status`: **전체 `needs_human_review`** (1,162/1,162건, 100%)
- `execution_verified`: **전체 `False`** (1,162/1,162건, 100%)

> [!IMPORTANT]
> **검토 상태와 실행 검증 상태의 독립성**:
> 본 데이터셋의 정답 라벨은 MCP 도구 명세 기반으로 작성된 초안이며, 실제 Hermes 런타임 환경에서 도구를 호출하여 응답을 검증한 결과(`execution_verified=True`)가 아니다. 향후 사람이 검토(`human_reviewed=True`)하더라도 실제 실행 성공으로 간주하거나 승격하지 않는다.

---

## 2. 재현 및 검증 방법 (Reproduction)

Python 3.10+ 표준 라이브러리만 사용하여 외부 의존성 없이 즉시 재현 가능하다.

```sh
# 1. 데이터 검토/감사 도구 실행 (콘솔 보고서 출력)
python3 apps/review/audit.py

# 2. JSON 요약 산출물 생성 (재독해 검증용)
python3 apps/review/audit.py --output-summary docs/review/audit_summary.json

# 3. dev-agy 감사 단위 테스트 실행 (7개 테스트)
python3 -m unittest discover -s tests/review -v

# 4. 공통 데이터셋 무결성 검증 스크립트 실행 (원본 보존 확인)
python3 scripts/validate_mcp_dataset.py
```

---

## 3. 핵심 감사 결과 (Detailed Audit Findings)

### [Finding 1] 보조 라벨 Action 클래스의 검증/평가 미등장 결함 (Critical Blocker)
- **영향 범위**: `labels.json` 및 `acceptable_choices` 내 `action` 보조 라벨
- **현황**: 9개 도구가 `action` enum을 정의(총 51개 action 정의)하며, 시나리오에서는 33개만 사용되고 18개는 미사용 상태다.
- **결함 내용**: `validation` 및 `test` 분할에서 정답으로 요구되는 action 중 일부가 **`train` 분할에 단 1건도 존재하지 않는 Zero-Shot 상황**이 발생한다.
  - `mcp__obsidian_vault__edit`: action `patch` (`obsidian_vault.edit.4` in val, `obsidian_vault.edit.5` in test) -> **train 등장 횟수: 0건** (train은 `append`, `at_line`, `window`만 등장)
  - `mcp__obsidian_vault__bases`: action `view` (`bases.4` in val) 및 `export` (`bases.5` in test) -> **train 등장 횟수: 0건** (train은 `list`, `query`, `read`만 등장)
  - `mcp__obsidian_vault__dataview`: action `status` (val) 및 `list` (test) -> **train 등장 횟수: 0건** (train은 `metadata`, `query`, `validate`만 등장)
  - `mcp__obsidian_vault__graph`: action `traverse` (val) 및 `statistics` (test) -> **train 등장 횟수: 0건** (train은 `backlinks`, `forwardlinks`, `path`만 등장)
- **위험**: 모델이 action 보조 분류를 수행할 경우, 검증 및 테스트셋에서 해당 action들은 학습 데이터 부재로 인해 0%의 재현율(Recall)을 기록하며 전체 모델 성능 지표를 심각하게 왜곡시킨다.
- **모델 담당자(`dev-claude`) 권고**:
  - 1차 기준 모델에서는 `tool` ID 예측만을 주 평가 지표로 삼고, `action` 예측은 분리 평가한다.
  - action 평가 시 train에 미등장한 zero-shot action 클래스는 마스킹하거나 제외하고 산출한다.

---

### [Finding 2] 평가셋(Val/Test) 미등장 도구 클래스 10개 (High Warning)
- **영향 범위**: 도구 커버리지 및 Macro-F1 지표
- **현황**:
  - 전체 등록 도구: 96개
  - Train 등장 도구: 96개 (100.0%)
  - Validation 등장 도구: 85개 (11개 미등장)
  - Test 등장 도구: 84개 (12개 미등장)
  - **Validation과 Test 양쪽 모두에 전혀 등장하지 않는 도구: 10개**
- **10개 미등장 도구 목록**:
  1. `mcp__grafana__get_alert_group`
  2. `mcp__grafana__get_current_oncall_users`
  3. `mcp__grafana__get_dashboard_summary`
  4. `mcp__grafana__get_oncall_shift`
  5. `mcp__grafana__get_snapshot`
  6. `mcp__grafana__list_provisioning_repositories`
  7. `mcp__obsidian_vault__list_resources`
  8. `mcp__obsidian_vault__workflow`
  9. `mcp__swallow_trader__swallow_reset_portfolio`
  10. `mcp__swallow_trader__swallow_trigger_pipeline`
- **원인 분석**: 해당 도구들의 시나리오는 `*.1` 접미사를 가진 단일 시나리오(또는 wording 변형 5개)로만 구성되어 있어, 시나리오 그룹 분할 격리 규칙에 의해 모든 사례가 `train` 분할에만 배정되었다.
- **위험**: 테스트셋 정확도 100%가 나오더라도 실제 96개 도구 전체를 판별할 수 있는지 검증된 것이 아니다.
- **권고**: 모델 평가 보고서 작성 시 단순 96개 클래스 기준 Macro-F1이 아닌, **"평가 가능 클래스(Val 85개, Test 84개)"에 대한 유효 커버리지**를 명시해야 한다.

---

### [Finding 3] 서버별 타겟 분포 급변 (Server Distribution Shift - High Warning)
- **영향 범위**: `select_tools`의 타겟 도구 서버 분포
- **실측 분포 통계**:

| MCP 서버 | Train 건수 (%) | Validation 건수 (%) | Test 건수 (%) | Val/Train 비율 |
|---|---:|---:|---:|---:|
| **grafana** | **154 (25.2%)** | **110 (46.6%)** | **108 (46.1%)** | **1.85배 급증** |
| swallow_trader | 186 (30.4%) | 48 (20.3%) | 50 (21.4%) | 0.67배 감소 |
| obsidian_vault | 140 (22.9%) | 34 (14.4%) | 32 (13.7%) | 0.63배 감소 |
| korean_law | 120 (19.6%) | 40 (16.9%) | 40 (17.1%) | 0.86배 감소 |
| kestrel | 12 (2.0%) | 4 (1.7%) | 4 (1.7%) | 0.86배 유사 |

- **원인**: Grafana 도구는 62개로 방대하여 분할별 1개씩 고르게 시나리오가 배정된 반면, Swallow와 Obsidian의 일부 도구는 Train에 5개씩 wording 변형이 집중되어 발생했다.
- **위험**: 학습 데이터의 사전 확률(Prior)을 강하게 학습하는 모델(예: Naive Bayes, n-gram, 빈도 기반 가중치)은 검증/테스트에서 Grafana 도구를 과소 예측(Under-prediction)하게 된다.

---

### [Finding 4] 결정 유형 극단적 불균형과 베이스라인 왜곡 (High Warning)
- **실측 분포 통계**:

| 결정 유형 (Decision) | Train 건수 (%) | Validation 건수 (%) | Test 건수 (%) | Overall 건수 (%) |
|---|---:|---:|---:|---:|
| **select_tools** | **600 (89.8%)** | **232 (93.5%)** | **230 (93.5%)** | **1,062 (91.4%)** |
| unsupported | 44 (6.6%) | 8 (3.2%) | 8 (3.2%) | 60 (5.2%) |
| clarify | 12 (1.8%) | 4 (1.6%) | 4 (1.6%) | 20 (1.7%) |
| no_tool | 12 (1.8%) | 4 (1.6%) | 4 (1.6%) | 20 (1.7%) |
| **합계** | **668 (100%)** | **248 (100%)** | **246 (100%)** | **1,162 (100%)** |

- **위험**:
  - 모델이 모든 입력을 무조건 `select_tools`로 예측하는 trivial majority 모델을 만들더라도 **전체 정확도 91.39% (Val/Test 93.5%)**를 달성한다.
  - 따라서 단순 Accuracy는 모델의 실제 의사결정 능력을 심각하게 과장(Inflation)한다.
- **권고**: 결정 유형 분류 지표는 전체 Accuracy 외에 **Macro-F1, Balanced Accuracy, 결정 유형별 Precision/Recall**을 필수로 보고해야 한다.

---

### [Finding 5] 후보 누락(unsupported) 편향 및 후보 길이 지름길 (Medium Warning)
1. **`missing_target_case`의 Swallow 편향**:
   - 도구가 후보 목록에서 제외되어 `unsupported`로 전이되는 20개 시나리오가 **100% Swallow Trader 도구에만 적용**되어 있다 (`swallow_reset_portfolio.1`, `swallow_trigger_pipeline.1`, `swallow_order.1..5`, `swallow_watchlist.1..5`).
   - Grafana, Korean Law, Obsidian Vault, Kestrel은 후보 목록에서 정답 도구가 빠져서 `unsupported`가 되는 사례가 전무하다.
   - 따라서 타 서버 도구 요청에서 필수 도구가 후보 목록에 없을 때 모델이 `unsupported`를 예측하지 못하고 엉뚱한 후보를 오선택(Hallucination)할 위험이 높다.
2. **후보 목록 길이 지름길(Heuristic Shortcut)**:
   - `candidate_tools`의 길이가 4인 사례는 100% `no_tool`, `clarify`, `unsupported`다.
   - `select_tools`는 최소 후보 길이가 5이며, 98%가 7, 8 또는 96이다.
   - 모델이 입력 텍스트와 후보 도구 명세를 대조하는 대신 단순 후보 개수(len==4)만 보고 비선택(Non-select)을 맞추는 지름길 학습(Shortcut learning)에 빠질 수 있다.

---

### [Finding 6] 문맥 의존성(Context) 및 Clarify 사례 극소 (Medium Warning)
- **전체 시나리오 중 선행 문맥(`context`) 보유**: 총 376개 시나리오 항목 중 **10개 시나리오(2.58% 행)**에 불과.
  - `korean_law.execute_tool.1..5`: 이전 도구 탐색 결과를 문맥으로 제공 (5개)
  - `clarify.1..5`의 `context_resolved` 변형: 모호한 대상을 이전 대화로 해소 (5개)
- **Clarify 시나리오 수**: 전체 5개 시나리오 (Train 3개, Val 1개, Test 1개).
  - Validation: `clarify.4` ("두 노트 사이의 링크 경로를 찾아줘.") 단 1개 시나리오 (4개 행).
  - Test: `clarify.5` ("그 주식의 가격 이력을 가져와줘.") 단 1개 시나리오 (4개 행).
  - Grafana, Korean Law, Kestrel에는 `clarify` 사례가 0건.
- **위험**: 문맥 유무에 따른 `clarify` vs `select_tools` 판별 능력이 충분히 검증되기 어렵다.

---

### [Finding 7] 어휘 중복 및 분할 격리 감사 (Pass with Minor Observations)
- **완전 일치 요청문 분할 누출**: 0건
- **정규화(구두점/공백 제거) 일치 분할 누출**: 0건
- **시나리오 ID 분할 격리**: 340개 시나리오 그룹 100% 격리 준수
- **유사 템플릿(Jaccard 0.45~0.58) 관측**: 16개 쌍 발견 (합성 데이터 생성 템플릿의 ID 치환 패턴: 예: `resource-alpha` vs `resource-delta`). 분할 누출 수준은 아니나 템플릿 암기로 인한 과적합 주의 필요.

---

## 4. 구체적 시나리오 ID 및 조치 가이드 (Actionable Checklist)

| 분류 | 관련 시나리오 ID | 발생 현상 | 권고 조치 방안 |
|---|---|---|---|
| **Action Blocker** | `obsidian_vault.edit.4`, `obsidian_vault.edit.5`, `obsidian_vault.bases.4`, `obsidian_vault.bases.5`, `obsidian_vault.dataview.4`, `obsidian_vault.dataview.5`, `obsidian_vault.graph.4`, `obsidian_vault.graph.5` | `patch`, `view`, `export`, `status`, `list`, `traverse`, `statistics` 액션이 train 미등장 | 기준 모델 평가 시 action 평가 분리 및 zero-shot 액션 마스킹 처리 |
| **Missing Tools** | `grafana.get_alert_group.1`, `grafana.get_current_oncall_users.1`, `grafana.get_dashboard_summary.1`, `grafana.get_oncall_shift.1`, `grafana.get_snapshot.1`, `grafana.list_provisioning_repositories.1`, `obsidian_vault.list_resources.1`, `obsidian_vault.workflow.1`, `swallow_trader.swallow_reset_portfolio.1`, `swallow_trader.swallow_trigger_pipeline.1` | 10개 도구가 train에만 존재하고 val/test에 미등장 | 평가 보고서에 "유효 평가 가능 도구 수(Val 85, Test 84)" 명시 |
| **Missing Target Bias** | `swallow_trader.swallow_order.1..5`, `swallow_trader.swallow_watchlist.1..5`, `swallow_trader.swallow_reset_portfolio.1`, `swallow_trader.swallow_trigger_pipeline.1` | 후보 누락 unsupported가 Swallow에만 편중 | 향후 v3 합성 시 타 서버(Grafana, Law 등)에도 missing_target_case 확장 요청 |
| **Multi Choice** | `swallow_trader.swallow_check_stock.1..5` | 5개 시나리오에서만 `swallow_check_stock`과 `swallow_get_stock_price` 복수 정답 허용 | 다중 정답 평가 시 Any-of-Acceptable 평가 로직 유지 |

---

## 5. 결론 및 다음 단계

1. **학습 모델 담당(`dev-claude`) 전달 사항**:
   - 도구 선택(`select_tools`)의 1차 목표인 96개 후보 대상 라우팅 모델 학습은 즉시 진행 가능.
   - 평가 시 **Decision Macro-F1**, **평가 가능 도구 84/85개 기준 Coverage**, **Zero-Shot Action 분리 집계**를 필수 적용할 것.
2. **데이터 및 통합 담당(`dev-codex`) 전달 사항**:
   - 차기 데이터셋(v3) 생성 시:
     - 10개 도구의 val/test 시나리오(`*.2`, `*.3`) 보강.
     - `action` enum의 split 간 고른 분포 배정.
     - Grafana, Korean Law에 대한 `missing_target_case` 및 `clarify` 시나리오 보강.
3. **dev-agy 산출물**:
   - `apps/review/audit.py`: 상시 재실행 가능한 stdlib 품질/커버리지 감사 도구.
   - `tests/review/test_audit.py`: 7개 단위 테스트(원본 보존, 플래그 분리, 지표 계산, CLI).
   - `docs/review/weekend-data-audit.md`: 본 감사 보고서.
   - `requirements-review.txt`: 표준 라이브러리 사용 확인 명세.
