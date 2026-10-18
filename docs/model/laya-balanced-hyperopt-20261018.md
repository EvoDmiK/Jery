# Laya 균형 학습·콜백·Hyperopt — 2026-10-18

이전 전체 데이터 실험의 과적합과 `window_none` 편향을 줄이기 위한 후속 실험이다. `dev-codex`에서 사용자의 명시적 요청으로 모델·실행기·설정을 함께 수정했다. 기존 실험의 완료 결과와 중단 체크포인트는 보존한다. 기존 Parent는 `0b6c6b0cb8324410a5866a7f48f6aa33`, 후속 [Parent는 `9325d1ca84944fe281f94b519c59f305`](https://mlflow.dove-nest.com/#/experiments/47/runs/9325d1ca84944fe281f94b519c59f305)이며 Experiment는 계속 `Jery`(47)다.

## 데이터와 균형

원본 `mcp-routing-v2`와 `decision-export-v1` 파일은 수정하지 않는다. 원본 668/248/246건과 전체 후보를 덮는 4,172/1,592/1,577개 창을 유지한다. 학습에서만 정답 bucket을 참조해 결정적 oversampling을 적용하며, 정답 필드를 모델 입력 특징으로 사용하지 않는다. 검증·평가에는 oversampling을 적용하지 않는다.

| 학습 bucket | 원래 창 | 에폭당 노출 | 비중 |
|---|---:|---:|---:|
| 도구 선택 | 607 | 3,412 | 40% |
| window_none | 3,409 | 3,412 | 40% |
| no_tool | 78 | 853 | 10% |
| clarify | 78 | 853 | 10% |

모든 원래 창을 최소 한 번 포함하고 bucket 안에서 균등하게 반복한 뒤 seed 3407로 섞는다. 총 8,530회 노출이며 micro batch 1, 누적 4로 에폭당 2,133 optimizer step이다. 반복 노출은 새로운 사례가 아니다. 특히 no_tool/clarify의 원본 사례가 각각 12건뿐이라는 한계가 남는다. 인덱스 전체와 실제 분포를 `balance.json`에 보존한다.

`clarify`는 필요한 대상·맥락이 부족하여 사용자 확인이 필요하다는 정답이다. 모델 확신이 낮다는 이유로 이 라벨을 만들지 않는다. `window_none`은 현재 창에 적합한 도구가 없다는 뜻이며 전체 후보에서 처리 불가인 `unsupported`와 구분한다. 기존 label-blind 다중 창 집계와 외부 예측 계약은 유지한다.

## 콜백 정책

- 최대 64에폭, 매 에폭 전체 validation 사례 평가.
- 감시값: 원본 사례 단위 `validation.case_accuracy`, 높을수록 좋음.
- `min_delta=0.004`: 검증 248건에서 한 사례가 개선되는 정도. 동률은 기존 체크포인트를 유지한다.
- `patience=5`: 유의한 개선 없이 5에폭 연속 진행하면 종료한다.
- 학습 전 epoch 0도 선택 후보다. 학습으로 악화되면 기본 head를 최종 모델로 복원할 수 있다.
- 최적 head는 별도 `best-epoch-N/`에 보존한다. 재시작용 optimizer·RNG 체크포인트는 최신 2개를 유지한다.
- 최종 선택 head를 다시 로드하고 validation 예측 및 확률 일치(차이 1e-6 이하), encoder 불변을 확인한 뒤 test를 1회 평가한다. test 결과는 콜백과 Hyperopt에 전달하지 않는다.
- NaN/Inf 학습 손실은 실패로 처리한다. 에폭·단계 로그를 즉시 flush하며, 재시작 시 콜백 상태를 검증 기록에서 재구성하고 중복 단계는 재생 구간에서 제거한다.

`report.selection`에 최대/완료/선택 에폭과 중단 이유를 구분한다. 학습 곡선은 실제 완료 에폭까지만 기록하며 마지막 곡선 값은 선택 모델 값과 다를 수 있다. 선택 모델의 검증 성능은 report와 Parent 요약을 따른다.

## Hyperopt

Hyperopt 0.2.7의 실제 `tpe.suggest`와 `Trials`로 5회 순차 탐색한다. 앞의 2회는 startup 탐색, 이후는 완료된 trial의 검증 결과를 이용하는 TPE다. 실패하거나 미완료인 trial을 성공 결과로 사용하지 않는다.

| 파라미터 | 범위 |
|---|---|
| head learning rate | log-uniform, 1e-5 ~ 1e-3 |
| weight decay | 0 / 0.01 / 0.05 |
| label smoothing | 0 / 0.05 / 0.1 |

목적함수는 `1 - best_validation_case_accuracy`다. 균형 비중·seed·콜백 정책은 고정하며 warmup은 64 step이다. 각 trial의 실제 Hyperopt document와 해석된 파라미터를 JSON에 저장하고, 재시작 시 완료 결과를 재구성한다. 저장된 pending trial의 파라미터는 다시 추첨하지 않는다. 최종 trial 선택은 검증 정확도, 동률이면 더 낮은 head LR이다. 평가 분할은 이전 실험에도 노출되었으므로 회귀 평가로만 해석한다.

MLflow Child에는 기존 핵심 4개 파라미터와 7개 메트릭만 기록한다. weight decay·label smoothing·balance·콜백·전체 Hyperopt 내역은 `configuration.json`, `details/balance.json`, `details/callback.json`, `hyperopt-trials.json` 등 JSON Artifact로 보관한다. 가중치는 safetensors 바이너리로 저장한다. Parent에는 원본 데이터와 균형 인덱스·검증 증거를 업로드했다.

## 한국어·영어 진단

`datasets/diagnostics/laya-ko-en-request-v1.json`의 validation 사례 12개를 대상으로 요청 문장만 번역했다. 후보 카드·순서·질문 스키마·정답은 동일하다. 번역은 사람이 검토하지 않은 초안이며, 후보 설명과 스키마에는 한국어가 남아 있으므로 완전한 영어 입력 비교는 아니다. 이 데이터와 결과는 학습·Hyperopt 목적함수에서 제외한다.

| 모델 | 한국어 요청 | 영어 요청 |
|---|---:|---:|
| 기본 head | 3/12 | 4/12 |
| 이전 LR 1e-4, 64에폭 head | 4/12 | 4/12 |

입력 잘림은 없었다. 이 작은 차이만으로 언어가 성능 저하의 주원인이라고 결론 내릴 수 없다. 후속 균형 모델의 성능 개선도 아직 입증되지 않았다. 결과 전체는 Parent의 `diagnostics/language-probe.json`에 있다.

## 실행·검증

```sh
DECISION_PYTHON=/Volumes/Dove-Nest-SSD/app-data/orca/kimdove/home/workspaces/Jery/main/tmp/venv-decision/bin/python
"$DECISION_PYTHON" scripts/run_laya_full_experiment.py \
  --python "$DECISION_PYTHON" \
  --state-dir outputs/laya-balanced-20261018/tracking \
  --model-config configs/model/laya-balanced-20261018.json --detach
```

설정은 `configs/model/laya-balanced-20261018.json`, 탐색 계획은 `configs/training/laya-balanced-20261018.json`이다. 상태·로그는 `outputs/laya-balanced-20261018/tracking/`, 모델별 결과는 `outputs/model/laya-balanced-20261018/`에 있다. 재시작 시 같은 명령을 사용하며 중복 실행은 파일 잠금으로 거부한다. 변경·가중치를 실행 중 덮어쓰지 않는다.

필수 의존성은 `requirements-decision.txt`에 기록했다. Hyperopt가 `pkg_resources`를 사용하므로 setuptools 80.9.0을 함께 고정했다. 학습 가상환경에만 설치했다.

작은 실제 MLX 검증은 원본 파생 train 8/val 3/test 3창, 균형화 후 20회 노출을 사용했다. 테스트 전용 patience 1/min_delta 1.0으로 첫 에폭 조기 종료와 epoch 0 복원을 강제했다. 중단 없이 실행한 경우와 첫 에폭 기록 직후 강제 중단·재시작한 경우 모두 5 step, 에폭 기록 1개, 최종 head SHA `378e43dafcdf0b5d2fa4bdc64db5cb4024fa313e77181b24dea78d942240bf96`, 복원 차이 0.0이었다. 이는 경로 검증이며 품질 측정이 아니다.

균형·콜백·추적 테스트와 실제 Hyperopt 제안/완료 결과 재구성 테스트를 실행했다. 데이터 v1/v2 검증, 요청 번역 쌍의 원본 보존 테스트도 통과했다. `git diff --check`와 `graphify update .`를 실행했다. 작업 브랜치는 `dev-codex`, 기준 HEAD `bd6d2ccd0e452deff84181748fc84e56df1ccdec`이며 이번 변경은 미커밋·미푸시 상태다.
