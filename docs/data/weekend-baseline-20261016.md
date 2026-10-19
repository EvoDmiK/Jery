# 2026-10-16 Mac mini M4 기준 학습 결과

등록 MCP 5개·도구 96개의 `mcp-routing-v2`로 첫 로컬 학습과 예측 검증을 완료했다. 모델 담당 Claude, 데이터 감사 담당 Antigravity, 데이터·계약·실행 통합 담당 Codex가 기존 담당 브랜치에서 함께 작업했다. 정답은 전부 미검토 합성이므로 아래 수치는 실제 사용 정확도가 아니다.

## 실행과 선택

- 환경: Mac mini M4, 메모리 32GB, arm64, Python 3.9.6, 외부 Python 의존성 없음.
- 데이터: `mcp-routing-v2`, 카탈로그 `hermes-mcp-57e701d87790`, train 668 / validation 248 / test 246.
- 명령: `PYTHONPATH=src python3 -m jery.model.cli --config configs/model/mcp-routing-v2-baseline.json --run-id v2-baseline-20261016 --workers 4`.
- 검색 기준 24개 + 선형 기준 24개 설정을 validation으로 비교했다. train만 fit/refit했고 선택 후 test를 보고했다.
- 선택: 선형 softmax, epochs 20, lr 0.3, l2 0, 도구 명세 특징 사용, seed 13.
- 실행 시간: 233.53초. worker 4개, worker당 스레드 1개. 학습 중에만 `caffeinate`로 절전을 방지했다.
- 초기 실험 코드: `adeb4af`. 이후 `c877fef`의 저장·복원 및 공개 예측 계약 수정은 기존 가중치와 선택 설정을 유지했다.

## 평가 결과

| 지표 | 검색 기준 | 학습한 선형 기준 |
|---|---:|---:|
| 도구 선택 정확도 | 59.13% | 82.61% |
| MCP 서버 정확도 | 88.70% | 99.13% |
| 결정 유형 정확도 | 90.65% | 98.37% |
| 결정 유형 macro F1 | 0.3160 | 0.8755 |
| 불필요 호출 비율 | 56.25% | 6.25% |

도구·서버 정확도는 정답이 `select_tools`인 230건에서 계산한다. test의 도구 클래스는 84개로 전체 등록 도구 96개를 모두 평가한 결과가 아니다. 불필요 호출 비율은 비선택 정답 16건 중 `select_tools`로 예측한 비율이다.

선형 모델의 도구를 맞힌 action 사례에서 보조 action 정확도는 36.67%다. action과 도구 정확도를 합쳐 보고하지 않는다. 정답이 도구 선택인 사례의 후보 순위 recall@1/3/5는 83.48% / 96.96% / 99.13%이며 후보 밖 선택은 0건이었다. 이미 메모리에 적재된 라우터의 p50 지연은 1.306ms, p95는 6.310ms다. 이 지연에는 프로세스 시작과 모델 파일 읽기가 포함되지 않는다.

결정별 test support는 `select_tools` 230, `no_tool` 4, `clarify` 4, `unsupported` 8로 비선택 유형의 평가 근거가 매우 적다. 새 표현·비Swallow 후보 누락·빈 후보에서 판단 편향과 오선택도 관측됐다. [모델 상세 보고](../model/baseline.md)와 [독립 데이터 감사](../review/weekend-data-audit.md)를 함께 확인한다.

## 저장·복원과 실제 분류 확인

모델 원본 및 결과는 Git에서 제외한 로컬 경로에 보존했다.

- `models/mcp-routing-v2/v2-baseline-20261016/linear.json.gz`
- `models/mcp-routing-v2/v2-baseline-20261016/retrieval.json.gz`
- `models/mcp-routing-v2/latest/model.json.gz`: 선택한 선형 모델의 동일한 바이트 사본.
- `outputs/model/mcp-routing-v2/v2-baseline-20261016/`: report.json, report.md, validation_sweep.json, test_predictions.jsonl.

전체/후보-only 카탈로그 cold load에서 학습 당시 TF-IDF 상태를 복원하며, 기존 validation 지표와 저장된 test 예측이 동일함을 검증했다. 통합 경로에 복사한 파일의 해시와 실험의 데이터 해시도 원본과 일치한다. 이 복원 검증은 설정 재선택이나 재학습이 아니다.

다음 명령으로 실제 저장 모델을 사용한다. MCP 호출은 실행하지 않는다.

```sh
python3 scripts/predict_tool.py --request '삼성전자 지금 주가 확인해줘.'
```

전체 96개 후보와 현재가 도구 하나만 있는 부분 후보 모두 `mcp__swallow_trader__swallow_get_stock_price`를 선택했고, 빈 후보에서는 `unsupported`와 빈 선택 목록을 반환했다. 사용자 정의 후보는 `--candidates FILE.json`, 이전 맥락은 `--context FILE.json`, 특정 모델은 `--model FILE.json.gz`로 전달한다.

현재 실험은 완료 상태다. 다음 실험은 [로컬 실행 절차](local-training.md)대로 별도 run ID를 사용한다. 추가 데이터 작업에는 검토된 정답, 더 다양한 비선택 요청, 비Swallow 후보 누락, action 보강과 별도 실제 요청 평가셋이 필요하다.
