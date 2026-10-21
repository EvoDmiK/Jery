# LR 단독 탐색 · 5-fold 검증

**진행 중인 실험:** [MLflow에서 보기](https://mlflow.dove-nest.com/#/experiments/47/runs/675bf807c9674425ba6f9de1ad536b91)

LR만 바꿔 성능을 비교한다. 나머지 설정은 `laya-ten-tools-train600-lr1e3-20261021` 실험이 최종 선택한 Child와 동일하게 고정했다.

> **상태 확인 — 2026-10-21 17:41 KST**
> 20개 조건 중 5개 완료. 6번째 조건의 두 번째 fold를 학습 중이었다. 이후 진행은 MLflow를 확인한다.

## 실험 설정

| 구분 | 설정 |
|---|---|
| 탐색 대상 | **LR `1e-3 ~ 5e-3`만 변경** |
| 탐색 방법 | Hyperopt TPE · 20개 조건 · 초기 탐색 6개 |
| 검증 | 동일한 층화 5-fold · 총 100회 학습 |
| 모델 | Laya multilingual · 인코더 고정, head 학습 |
| 환경 | Linux · RTX 5070 Ti |

### 고정 파라미터

| 파라미터 | 값 |
|---|---:|
| Warmup steps | 10 |
| Gradient clipping (`max_grad_norm`) | 1.0 |
| Scheduler | `linear` |
| Weight decay | 0.05 |
| Label smoothing | 0.1 |
| 유효 배치 / 누적 횟수 | 4 / 1 |
| Seed | 3407 |
| 최대 에폭 | 32 |
| 조기 종료 | 최소 8에폭 관찰 · patience 6 · min_delta 0.004 |

기준 Child는 `c04cdad0246b4d6896ef89e5f9032ce2`이며, 이전 실험의 검증 정확도는 0.95였다. 이 값은 **이번 5-fold 결과가 아니다.** 기준 선택에는 test 점수를 사용하지 않았다.

전체 원본 설정은 [학습 계획의 `reference_model_config`](../../configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-20261021.json)에 보존했다. 실행 전 LR과 결과 저장 경로를 제외한 모든 모델 설정이 기준과 같은지 검사한다.

## 데이터와 검증

| 항목 | 구성 |
|---|---|
| 데이터 | `hermes-ten-tools-v3` · 합성 학습 사례 600개 |
| Fold별 구성 | 학습 **480개** / 검증 **120개** |
| 클래스 균형 | 검증에 도구별 **12개** |
| 그룹 보존 | 같은 `scenario_id`는 학습·검증에 걸쳐 나누지 않음 |
| 외부 데이터 | 기존 validation 20개는 CV에 사용하지 않음 · test 20개는 별도 평가 |

모든 LR 조건이 같은 fold를 사용한다. Test는 fold 구성·체크포인트 선택·LR 탐색에 사용하지 않는다. 합성 데이터의 반복 실험이므로 실제 사용자 요청에 대한 성능과는 구분한다.

## MLflow 읽는 법

구조는 **전체 실험 → LR 조건 20개 → 조건별 fold 5개**다.

- **학습 진행:** 조건을 펼쳐 fold 행의 에폭 지표를 확인한다.
- **조건 비교:** 5개 fold가 끝나면 validation 평균과 표준편차가 표시된다.
- **탐색 기준:** `cv_search_accuracy` — fold별 best-trained validation 정확도의 평균.
- **Test 결과:** 5개 fold 모델의 평균이다. 전체 600개로 학습한 단일 모델의 점수는 아니다.

초기 head 보호로 인해 `val_accuracy`와 `cv_search_accuracy`가 다를 수 있다. 지표의 정확한 정의는 [교차 검증 계약](../contracts/laya-cross-validation-v1.md)에 정리했다.

## 실행 자료

| 자료 | 위치 |
|---|---|
| 학습 계획 | [LR 탐색·fold 설정](../../configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-20261021.json) |
| 모델 설정 | [고정 파라미터](../../configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-model-20261021.json) |
| 실행기 | [CV supervisor](../../scripts/run_laya_cv_experiment.py) |
| 데이터 분할 | [층화·그룹 보존 fold builder](../../scripts/build_laya_stratified_folds.py) |
| 상태 파일 | `outputs/laya-ten-tools-lr-only-cv5-lr1e3ref-20261021/tracking/state.json` |

실행 코드 기준은 `4a98932`다. 문서에 적힌 상태는 확인 시점의 기록이며, 실행 중인 상태 파일과 MLflow가 최신 진행 상황을 제공한다.

### 확인한 항목

- 기준 Child와 LR 외 설정 전체 일치.
- 실제 Hyperopt 후보 8개에서 비-LR 설정 불변.
- CV 통합 테스트 4개 및 기존 데이터 검증 통과.
- 실제 CUDA 학습, MLflow 에폭 동기화, 비교 파라미터 10개 기록 확인.

이전 중단 실험과 삭제된 MLflow 항목은 재개하거나 복원하지 않았다.
