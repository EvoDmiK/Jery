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

## 예측 함수

```python
import json
from jery.model import predict   # PYTHONPATH=src

catalog = json.load(open('datasets/mcp-routing-v2/tools.json'))
out = predict(
    request='Swallow 가상 포트폴리오 보유 종목 보여줘',
    context=[{'role': 'user', 'content': '...'}],      # 도구 선택 이전 대화만
    candidate_tools=['mcp__swallow_trader__swallow_get_portfolio', 'mcp__grafana__list_datasources'],
    catalog=catalog,               # tools.json dict, 명세 list 또는 {id: spec}
    model=None,                    # None → $JERY_MODEL_PATH → models/mcp-routing-v2/latest/model.json.gz
)
# {'decision': 'select_tools',
#  'acceptable_choices': [{'tool': 'mcp__swallow_trader__swallow_get_portfolio'}],   # action은 학습된 경우만
#  'ranked_tools': [{'tool': ..., 'score': ...}, ...]}                               # 후보만, 점수 내림차순
```

`decision`은 `select_tools`, `no_tool`, `clarify`, `unsupported` 중 하나다. `acceptable_choices`는 `select_tools`일 때만 원소 1개이며 항상 `candidate_tools` 안에 있다. linear의 `score`는 결정 선택지를 포함한 softmax 확률, retrieval은 코사인이다. `model`에는 로드한 라우터 객체나 경로를 넘길 수 있고, 경로별로 프로세스 안에서 캐시한다.

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

## 한계

- 미검토 합성 데이터이며 같은 기능군·표현 패턴이 분할 간 반복된다. test가 validation보다 높은 것은 비선택 결정 support가 4/4/8건뿐인 소표본 변동으로 보이며 일반화 근거가 아니다.
- 선택 설정이 sweep 경계(epochs·lr 최대값)에 있다. 더 넓은 sweep은 새 run에서 validation으로만 해야 하며, 이번 test 결과를 보고 재튜닝하지 않는다.
- Grafana 명세가 영어라 한국어 요청과의 TF-IDF 유사도가 낮다. retrieval 기준은 이름·영문 키워드 일치에 크게 의존하고 `no_tool`·`clarify`를 예측할 수 없다.
- linear는 도구별 가중치를 쓰므로 train에 없던 새 도구는 명세 특징(`pair:*`)과 서버 가중치만으로 점수화된다. 카탈로그가 바뀌면 재학습과 재평가가 필요하다.
- action은 보조 출력이며 정확도가 낮다. 다른 인자 생성은 범위 밖이다. 지연은 단일 프로세스 Python 측정값이며 서버 오버헤드를 포함하지 않는다.
- 더 큰 모델(임베딩·소형 LM)은 아직 비교하지 않았다. 도입 시 같은 CLI 프로토콜과 지연·자원 기록으로 이 기준과 비교한다.
