# LR 고정 · 학습 파라미터 탐색

지정한 run의 설정으로 첫 조건을 학습하고, 나머지 19개 조건은 LR을 고정한 채 다른 학습 파라미터를 탐색한다. 기존과 동일한 층화·시나리오 그룹 보존 5-fold를 사용한다.

[새 MLflow 실험](https://mlflow.dove-nest.com/#/experiments/47/runs/d9636e10f0b6449dbf6975e945e07aad) · 실행 코드 `94f6364`

## 기준 모델

[지정 run](https://mlflow.dove-nest.com/#/experiments/47/runs/976cdae4215f4bf18175187fee21a2ce) · `laya-ten-tools-train600-refine20-lr0.000288845`

| 파라미터 | 첫 조건 |
|---|---|
| Head LR | **0.00028884521694941566** · 전 조건 고정 |
| 유효 배치 | 4 · micro batch 4, accumulation 1 |
| Warmup | 10 optimizer steps |
| Clipping | 1.0 |
| Scheduler | linear |
| Weight decay | 0.1 |
| Label smoothing | 0.0 |
| Seed | 3407 · 전 조건 고정 |
| 종료 정책 | 최대 32에폭 · 최소 8에폭 · patience 6 · min_delta 0.004 |

전체 기준 설정은 MLflow의 실제 `configuration.json`에서 가져왔다. 첫 조건은 결과 저장 경로와 CV 분할 경로 외 모델 설정을 그대로 사용한다. 지정 run의 학습된 head에서 이어 학습하지 않고, 모든 조건이 동일한 고정 base에서 새로 시작한다.

## 나머지 19개 조건

Hyperopt TPE를 사용한다. 첫 기준 조건도 탐색 이력에 포함하며 초기 탐색 수는 6개다. 후속 조건에서 기준 조합 그대로의 반복은 제외한다.

| 탐색 항목 | 후보 |
|---|---|
| 유효 배치 | 4, 8, 16 · accumulation 1, 2, 4 |
| Warmup steps | 0, 10, 50, 100 |
| Gradient clipping | 0.5, 1.0, 2.0 |
| Scheduler | linear, cosine |
| Weight decay | 0.0, 0.05, 0.1, 0.2 |
| Label smoothing | 0.0, 0.05, 0.1 |

Seed·학습 상한·콜백·인코더 고정 정책·입력 형식은 유지한다. 비선택 클래스가 없는 현재 과제에서 `window_none_loss_weight`는 탐색하지 않는다.

## 데이터와 MLflow

- 합성 train 600개 → fold마다 학습 480개 / 검증 120개.
- 조건 20개 × fold 5개 = **총 100회 학습**. 원래 validation 20개는 사용하지 않는다.
- MLflow 구조는 **Parent → 조건 20개 → 조건별 fold 5개**.
- 탐색은 fold별 best-trained validation accuracy의 평균으로 결정한다. Test는 선택에 사용하지 않는다.
- Trial의 test 점수는 5개 fold 모델의 평균이며, 전체 600개로 학습한 단일 모델의 점수가 아니다.
- 핵심·비교 파라미터 10개와 필수 지표를 표에 남기고, 전체 설정과 상세 이력은 JSON Artifact에 저장한다.

과거 test 결과를 반복 관찰한 데이터와 사용자 지정 기준을 활용한 탐색이다. 실제 사용 성능이나 독립적인 최종 평가로 해석하지 않는다.

## 실행 자료

- [학습 계획](../../configs/training/laya-ten-tools-fixedlr-params-cv5-20261022.json)
- [기준 모델 설정](../../configs/training/laya-ten-tools-fixedlr-params-cv5-20261022-model.json)
- [CV 실행기](../../scripts/run_laya_cv_experiment.py)
- [교차 검증 계약](../contracts/laya-cross-validation-v1.md)

```sh
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src \
  /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_cv_experiment.py \
  --plan configs/training/laya-ten-tools-fixedlr-params-cv5-20261022.json \
  --model-config configs/training/laya-ten-tools-fixedlr-params-cv5-20261022-model.json \
  --state-dir outputs/laya-ten-tools-fixedlr-params-cv5-20261022/tracking \
  --python /home/persona-ai/envs/jery-cuda/bin/python --detach
```

## 검증 결과

서버에서 CV 5개·tracking 15개·계획 6개 테스트가 모두 통과했다. 실제 Hyperopt 후보 20개를 생성해 다음을 확인했다.

- 첫 후보가 지정 run의 설정과 일치.
- 나머지 19개 후보도 LR이 정확히 동일하며 기준 조합 전체의 반복은 제외.
- 유효 배치와 실제 학습 override·MLflow 값 일치.
- Test 값을 바꿔도 다음 후보가 동일.

기존 데이터 검증도 통과했다. 추가 의존성은 없다.

## 실행 확인

2026-10-22 12:37 KST 기준, 첫 조건의 fold 1에서 실제 CUDA 1에폭과 MLflow 동기화를 확인했다. 기준 run과 실제 첫 조건의 전체 모델 설정을 비교해 CV 분할·저장 경로 외 일치를 확인했고, MLflow의 LR·weight decay·label smoothing도 원본 값과 일치했다. 전체 20개 조건 완료를 의미하지 않는다.
