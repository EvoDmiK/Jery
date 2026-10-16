# MCP v2 기준 라우터

`datasets/mcp-routing-v2`(카탈로그 `hermes-mcp-57e701d87790`, 서버 5개·도구 96개)를 읽는 로컬 기준 모델 두 개와 학습·평가 CLI다. Python 3.9+ 표준 라이브러리만 사용하며 모델 다운로드나 외부 서비스가 없다(`requirements-model.txt`).

> 결과는 미검토 합성 정답(`needs_human_review`, `execution_verified: false`)에 대한 **탐색 실험**이다. 실제 사용 정확도나 검토된 정답 성능으로 인용하지 않는다.

## 모델

입력은 `model_input(row)`가 만드는 `request`, `context`, `candidate_tools`와 카탈로그의 후보 명세(서버, 이름, 설명, 입력 스키마 속성)뿐이다. `id`, `scenario_id`, `split`, `source`, `review_status`, `acceptable_choices`는 특징에 쓰지 않는다. 텍스트 특징은 단어 unigram과 단어 내부 문자 2·3-gram이다(한국어 형태소 분석기 없음).

- **retrieval**: 후보 명세와 요청·맥락의 문자 n-gram TF-IDF 코사인으로 후보를 정렬한다. IDF는 카탈로그 명세에서만 계산한다. 최고 점수가 임계값 미만이면 fallback 결정을 반환한다. 임계값·fallback은 validation으로 고른다. 라벨로 학습하지 않는 기준선이다.
- **linear**: 후보 도구와 `no_tool`·`clarify`·`unsupported` 세 가상 선택지에 대한 softmax다. 도구 점수는 도구별·서버별 가중치와 요청 특징의 내적, 그리고(`use_spec`) 명세 코사인·도구명 일치·서버명 언급 특징의 합이다. 결정 선택지는 요청 특징과 후보 집합 통계(빈 후보, 후보 수, 최대 명세 유사도)를 본다. 필요한 도구가 후보에서 빠지면 `unsupported`가 이길 수 있다. 여러 `acceptable_choices`는 대안이므로 정답 확률 합을 최대화한다. action이 여러 개 학습된 도구는 별도 softmax로 action을 고른다. train에서만 SGD로 학습한다.

두 모델 모두 후보에 있는 도구만 점수화하므로 후보 밖 선택이 구조적으로 불가능하며, `predict`가 다시 확인한다. 빈 후보는 `select_tools`를 반환하지 않는다.

## 실행

```sh
PYTHONPATH=src python3 -m jery.model.cli --config configs/model/mcp-routing-v2-baseline.json [--run-id ID] [--workers N]
python3 -m unittest discover -s tests/model -v
python3 -m unittest discover -s tests/evaluation -v
```

한 명령이 다음을 수행한다.

1. `manifest.json`의 SHA-256과 각 파일을 대조하고 다르면 중단한다. 데이터는 수정하지 않는다.
2. validation sweep: retrieval 24개(임계값 8 × fallback 3), linear 24개(epochs {5,10,20} × lr {0.1,0.3} × l2 {0,1e-4} × use_spec). linear는 spawn 프로세스 최대 `max_workers`(기본 4, 각 1스레드)로 병렬 실행한다.
3. 기준: validation `joint_accuracy`, 동률이면 decision macro F1, 도구 정확도. 각 계열의 최적 설정을 고정하고 linear는 train만으로 다시 학습한다(시드 고정, 결정적).
4. test를 한 번 평가해 보고한다. 같은 `run_id`가 있으면 실행을 거부한다. test 결과로 설정을 바꾸는 경로는 없다.

`configs/model/smoke.json`은 약 10초짜리 동작 확인용이다.

산출물은 Git에서 무시되는 경로에 저장한다.

- `outputs/model/mcp-routing-v2/<run_id>/report.json`, `report.md`: 지표, 설정, 데이터 해시, 시드, 명령, git SHA, 자원과 환경.
- `validation_sweep.json`: 모든 validation 시도. `test_predictions.jsonl`: 사례별 test 예측과 top-5.
- `models/mcp-routing-v2/<run_id>/{linear,retrieval}.json.gz`와 선택 모델 사본 `models/mcp-routing-v2/latest/model.json.gz`.

## 예측 함수 (prediction-v1)

공개 함수는 [`docs/contracts/prediction-v1.md`](../contracts/prediction-v1.md)를 따른다. 입력은 `jery.runtime.prediction.model_input` whitelist를 거쳐 후보 명세(`id`, `server`, `name`, `description`, `input_schema`)만 담긴 catalog로 축소되고, 출력은 반환 전에 `validate_prediction`으로 검사한다.

```python
import json
from jery.model import predict   # PYTHONPATH=src

catalog = json.load(open('datasets/mcp-routing-v2/tools.json'))
out = predict(
    request='옵시디언 볼트에서 회의록 노트를 찾아줘',              # 비어 있지 않은 문자열
    context=[],                                                   # [{'role': 'user'|'assistant'|'tool', 'content': str}]
    candidate_tools=['mcp__obsidian_vault__vault', 'mcp__grafana__query_loki_logs'],   # 중복 없는 ID, 빈 목록 허용
    catalog=catalog,               # tools.json dict (spec list / {id: spec}도 {'tools': [...]}로 변환)
    model=None,                    # None → $JERY_MODEL_PATH → models/mcp-routing-v2/latest/model.json.gz
)
# {'decision': 'select_tools',
#  'choices': [{'tool': 'mcp__obsidian_vault__vault', 'action': 'search'}],   # select_tools일 때만 1개
#  'ranked_tools': ['mcp__obsidian_vault__vault', 'mcp__grafana__query_loki_logs'],  # 후보 ID만
#  'confidence': 0.86}                                                       # linear만, 보정되지 않은 softmax 값

# 계약 helper와 그대로 호환된다.
from jery.runtime.prediction import model_input
features = model_input(example, catalog)   # example의 라벨·ID·출처는 버려진다
out = predict(**features)
```

- 후보가 catalog에 없거나 중복이면 `ValueError`. 후보 밖 도구, 스키마 `action.enum` 밖 action, 잘못된 순위를 router가 내면 `ValueError`로 거부한다. action은 학습된 action 중 해당 후보 스키마 enum에 있는 값만 낸다.
- 내부 router와 `jery.evaluation.metrics`는 `acceptable_choices`와 점수 포함 `ranked_tools`([{tool, score}]) 형식을 유지한다. `jery.model.predict.to_contract`가 경계에서 변환한다.
- `model`에는 로드한 router나 artifact 경로를 넘길 수 있다. 파일 캐시는 (실제 경로, inode, 크기, mtime) 기준이라 artifact를 교체하면 다시 로드한다. catalog는 캐시 키가 아니며 매 호출 입력으로만 쓴다.

### 저장 상태와 cold load

학습 때의 TF-IDF 상태(`idf`, `default_idf`, `fit_tools`, `fit_catalog_sha256`)를 artifact의 `index`에 저장하고 로드 시 그대로 복원한다. 예측 시 받은 catalog로 IDF를 다시 fit하지 않는다. 후보 명세 벡터는 받은 명세와 고정 IDF로 계산한다. 그래서 전체 catalog와 후보만 담긴 catalog로 cold load해도 같은 요청·후보의 점수와 예측이 같다(`tests/model`의 `test_cold_load_with_subset_catalog_matches_training_scores`). `index`가 없는 구 artifact는 로드를 거부한다.

이전 버전은 로드할 때 넘겨받은 catalog로 index를 다시 fit했다. `predict`가 후보만 넘기면 IDF가 학습 때와 달라져 점수가 바뀌었다(`test_refit_on_subset_would_change_scores`). 2026-10-16 baseline artifact는 가중치·설정·action을 바꾸지 않고, 학습 때 쓴 원본 `tools.json`(SHA-256 `d8a97f14…`, artifact meta와 일치 확인)으로 index 상태만 보강했다.

```sh
PYTHONPATH=src python3 -m jery.model.augment_index --catalog datasets/mcp-routing-v2/tools.json \
  models/mcp-routing-v2/v2-baseline-20261016/linear.json.gz models/mcp-routing-v2/v2-baseline-20261016/retrieval.json.gz
cp models/mcp-routing-v2/v2-baseline-20261016/linear.json.gz models/mcp-routing-v2/latest/model.json.gz
```

| artifact | 보강 전 SHA-256 | 보강 후 SHA-256 |
|---|---|---|
| `v2-baseline-20261016/linear.json.gz` (= `latest/model.json.gz`) | `f56a4fff…a659b9` | `25143fa2…68df6` |
| `v2-baseline-20261016/retrieval.json.gz` | `6d1ee97f…4d69ae` | `a8c50254…391ff` |

원본은 같은 위치의 `*.pre-index`로 보존했다(Git 무시 경로). 보강한 두 artifact를 새 코드로 로드했을 때 validation·test 지표가 `report.json`과 완전히 같았고, test 예측(decision, 선택, top-5)이 저장된 `test_predictions.jsonl`과 모두 같았다. 공개 `predict`(후보만 catalog) 결과도 전체 catalog 결과와 같았다. 재학습이나 test 기반 재설정은 하지 않았다.

## 지표

`src/jery/evaluation/metrics.py`가 계산한다.

- `joint_accuracy`: 결정이 맞고, `select_tools`면 첫 선택이 허용 대안 중 하나인 비율.
- `tool_accuracy`, `server_accuracy`: 정답이 `select_tools`인 사례에서 선택 도구·서버가 맞는 비율.
- `per_decision`: 결정별 precision/recall/F1/support. 예측이 없으면 precision은 `null`.
- `unnecessary_call_rate`: 정답이 `no_tool`·`clarify`·`unsupported`인 사례 중 `select_tools`로 예측한 비율.
- `candidate_recall_at_k`: 정답이 `select_tools`인 사례에서 `ranked_tools` 상위 k에 허용 도구가 있는 비율(k=1,3,5).
- `out_of_candidate_selections`: 후보 밖 선택 건수(항상 0이어야 함). `action_accuracy_given_tool`: 도구가 맞고 정답 action이 있는 사례의 보조 지표.

## 2026-10-16 실행 결과 (`v2-baseline-20261016`)

코드 `adeb4af`, 시드 13, Apple M4 32GB(10코어), macOS, Python 3.9.6, 4 프로세스 × 1스레드. 전체 233.5초(워커 CPU 756.8초, 최종 학습 37.6초), 최대 RSS 프로세스당 약 263MB. linear 가중치 630,528개, 모델 파일 7.4MB(gzip JSON). 데이터 해시는 manifest와 일치했다(`train.jsonl` `5eb09bc5…`, `validation.jsonl` `6840efb9…`, `test.jsonl` `7a1fc560…`, `tools.json` `d8a97f14…`).

선택 설정: linear `epochs=20, lr=0.3, l2=0, use_spec=true` (validation joint 0.7298). retrieval 최적은 `threshold=0.04, fallback=unsupported`.

| 모델 | split | joint | decision acc | macro F1 | tool acc | server acc | 불필요 호출률 | R@1 | R@3 | R@5 | 지연 p50 / p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retrieval | validation | 0.601 | 0.915 | 0.321 | 0.625 | 0.910 | 0.500 | 0.629 | 0.832 | 0.901 | |
| retrieval | test | 0.569 | 0.907 | 0.316 | 0.591 | 0.887 | 0.563 | 0.596 | 0.791 | 0.878 | 0.19 / 0.71 |
| linear | validation | 0.730 | 0.964 | 0.680 | 0.737 | 0.983 | 0.188 | 0.750 | 0.927 | 0.970 | |
| linear | test | 0.829 | 0.984 | 0.876 | 0.826 | 0.991 | 0.063 | 0.835 | 0.970 | 0.991 | 1.31 / 6.31 |

test 결정별 precision / recall (support):

| 모델 | select_tools | no_tool | clarify | unsupported |
|---|---|---|---|---|
| retrieval | 0.961 / 0.952 (230) | – / 0.0 (4) | – / 0.0 (4) | 0.222 / 0.5 (8) |
| linear | 0.996 / 0.991 (230) | 1.0 / 0.5 (4) | 1.0 / 1.0 (4) | 0.727 / 1.0 (8) |

linear test action 정확도(도구 정답 조건)는 0.367이다. test에 정답으로 나타나는 도구는 84개로, 96개 전체 클래스 성능이 아니다. `use_spec=false` 최적은 validation joint 0.690으로, 명세 특징이 약 +0.04 기여했다.

## 데이터 분할 특성

`PYTHONPATH=src python3 -m jery.model.profile`로 재현한다(분석용이며 특징으로 쓰지 않는다). 새 CLI 실행은 `report.json`의 `data.profile`에 같은 내용을 기록한다.

| 항목 | train | validation | test |
|---|---:|---:|---:|
| `select_tools` / `unsupported` / `no_tool` / `clarify` | 600 / 44 / 12 / 12 | 232 / 8 / 4 / 4 | 230 / 8 / 4 / 4 |
| 정답으로 나타나는 도구 | 96 | 85 | 84 |
| `select_tools` 중 Grafana 비중 | 25.7% | 47.4% | 47.0% |
| `unsupported` 중 후보 제거형(전부 Swallow) / 카탈로그 밖 요청 | 32 / 12 | 4 / 4 | 4 / 4 |
| train에 없던 (도구, action) 사례 | – | 16 (4종) | 16 (4종) |

- **train 전용 도구 10개**: grafana `get_alert_group`, `get_current_oncall_users`, `get_dashboard_summary`, `get_oncall_shift`, `get_snapshot`, `list_provisioning_repositories`, obsidian_vault `list_resources`, `workflow`, swallow_trader `swallow_reset_portfolio`, `swallow_trigger_pipeline`. validation·test 지표는 이 도구들을 평가하지 않는다.
- **unseen action**: validation `graph:traverse`, `dataview:status`, `edit:patch`, `bases:view`, test `dataview:list`, `bases:export`, `graph:statistics`, `edit:patch`는 train에 없다. linear는 학습한 action만 내므로 이 사례의 action은 구조적으로 틀린다. test action 정확도 0.367의 주요 원인이다.
- **Grafana 비중 이동**: Grafana가 train 선택 사례의 26%, validation·test의 47%다. 전체 정확도는 Grafana 성능에 크게 좌우되며, train 분포와 다른 평가 분포다.
- **Swallow 전용 missing-target 편향**: "필요한 도구가 후보에서 빠져 `unsupported`"인 사례는 모두 Swallow(초기화·주문·파이프라인·관심종목)다. 다른 서버에서 정답 도구가 빠진 경우의 `unsupported` 판단은 학습·평가되지 않았다. 아래 smoke에서 Grafana·법령 요청에 정답 도구 없는 부분 후보를 주면 `unsupported`로 가지만, 정답이 있어도 틀릴 수 있다.
- **null support 부족**: validation·test의 `no_tool`·`clarify`가 각 4건, `unsupported` 8건이다. 결정별 precision/recall은 1건 차이로 0.25씩 움직인다. train에도 빈 후보 사례가 없어 빈 후보는 요청과 무관하게 `unsupported`로 기운다.

## 선택 artifact smoke (새 한국어 요청, 정답 없음)

`models/mcp-routing-v2/latest/model.json.gz`(linear, SHA-256 `25143fa2…`)를 새 프로세스에서 로드해 데이터셋에 없는 요청을 전체 96개 / 부분(`grafana__query_loki_logs`, `obsidian_vault__vault`, `korean_law__search_law`) / 빈 후보로 실행했다. 모든 출력이 `validate_prediction`을 통과했고 후보 밖 선택은 없었다. 결과로 설정을 바꾸지 않았다. 아래 기대값은 사람의 판단이다.

| 요청 | 전체 후보 | 부분 후보 | 빈 후보 |
|---|---|---|---|
| Grafana에서 지난 한 시간 동안 에러 로그가 많은 서비스 알려줘 | `get_assertions` (0.06, 오답 가능성 큼) | `unsupported` (Loki 도구가 있는데 놓침) | `unsupported` |
| 옵시디언 볼트에서 회의록 노트를 찾아줘 | `vault` / `search` | `vault` / `search` | `unsupported` |
| 주택임대차보호법 최신 조문 확인해줘 | `no_tool` (오답) | `unsupported` (오답) | `unsupported` |
| 오늘 저녁 메뉴 추천해줘 | `no_tool` | `unsupported` (오답) | `unsupported` (`no_tool`이 맞음) |
| 그거 다시 실행해줘 | `clarify` | `unsupported` | `unsupported` |
| Swallow 가상 계좌 초기화해줘 | `swallow_reset_portfolio` (0.97) | `unsupported` | `unsupported` |

합성 템플릿 밖 표현과 부분 후보에서는 `unsupported` 쪽으로 치우친다. 실제 사용 전 검토된 실제 요청 평가와 빈·부분 후보 학습 사례 보강이 필요하다.

## 한계

- 미검토 합성 데이터이며 같은 기능군·표현 패턴이 분할 간 반복된다. test가 validation보다 높은 것은 비선택 결정 support가 4/4/8건뿐인 소표본 변동으로 보이며 일반화 근거가 아니다. 분할 간 Grafana 비중·unseen action·Swallow 전용 missing-target 편향은 위 [데이터 분할 특성](#데이터-분할-특성)을 본다.
- 선택 설정이 sweep 경계(epochs·lr 최대값)에 있다. 더 넓은 sweep은 새 run에서 validation으로만 해야 하며, 이번 test 결과를 보고 재튜닝하지 않는다.
- Grafana 명세가 영어라 한국어 요청과의 TF-IDF 유사도가 낮다. retrieval 기준은 이름·영문 키워드 일치에 크게 의존하고 `no_tool`·`clarify`를 예측할 수 없다.
- linear는 도구별 가중치를 쓰므로 train에 없던 새 도구는 명세 특징(`pair:*`)과 서버 가중치만으로 점수화된다. 카탈로그가 바뀌면 재학습과 재평가가 필요하다.
- action은 보조 출력이며 정확도가 낮다. 다른 인자 생성은 범위 밖이다. 지연은 단일 프로세스 Python 측정값이며 서버 오버헤드를 포함하지 않는다.
- 더 큰 모델(임베딩·소형 LM)은 아직 비교하지 않았다. 도입 시 같은 CLI 프로토콜과 지연·자원 기록으로 이 기준과 비교한다.
