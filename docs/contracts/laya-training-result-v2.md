# Laya 학습 결과 v2

`training_policy_version: laya-routing-v2`인 CUDA 실험에 적용한다. 데이터 계약 `decision-export-v1`, 원본 `mcp-routing-v2`와 공개 prediction 계약은 바꾸지 않는다. 버전 필드가 없는 기존 결과는 legacy로 보존한다.

## Loss와 입력

원래 모든 학습 창과 기존 train-only 균형 인덱스를 유지한다. `window_none_loss_weight`는 학습 정답이 `window_none`인 행의 loss에만 적용한다. 다른 행은 가중치 1이다. 가중 loss는 sampled train 전체의 평균 가중치로 정규화한다. 각 microbatch의 가중치 합으로 나누지 않으므로 microbatch 1에서도 효과가 사라지지 않는다.

정답·가중치·출처·분할은 모델 특징이 아니다. collator가 받은 loss 가중치는 forward 전에 분리하며 추론 입력이나 집계에는 전달하지 않는다. test·validation으로 학습 정규화 통계를 계산하지 않는다.

## 세 가지 모델

- **Baseline:** 학습 전 epoch 0.
- **Best trained:** epoch 1 이상에서 validation case accuracy가 실제로 가장 높은 head. 동률이면 먼저 나온 epoch를 유지한다. 작은 개선도 저장한다.
- **Serving selected:** best trained가 baseline보다 높으면 해당 head, 그렇지 않으면 baseline. 최종 test는 이 head를 평가한다.

`min_delta`는 trained-only 조기 종료의 개선 판단에만 사용한다. 최고 학습 head 저장이나 baseline 보호 기준에 적용하지 않는다. `min_epochs` 이전에는 bad epoch를 누적하지 않는다. 초기 제어 실험은 정확히 10에폭, 일반 학습은 최대 64에폭을 지원한다.

## JSON 필드

`report.selection`의 기존 `best_epoch`, `best_score`, `selected_epoch`는 serving-selected head를 뜻한다. `completed_epochs`, `max_epochs`, `reason`은 기존 의미를 유지한다. 다음 필드를 추가한다.

| 필드 | 의미 |
|---|---|
| `policy_version` | `laya-routing-v2` |
| `best_trained_epoch` | 실제 학습한 최고 epoch, 1 이상 |
| `best_trained_score` | 해당 epoch의 validation case accuracy |
| `search_score` | `best_trained_score`와 같음. baseline으로 올림하지 않음 |
| `trained_bad_epochs` | trained-only 조기 종료의 미개선 횟수 |
| `min_epochs` | 조기 종료 전 관찰 예산 |

`report.best_trained`는 `artifact`, `artifact_sha256`, `epoch`, `validation_case_accuracy`, `format`을 포함한다. `report.last_trained`와 `report.restore`는 각각 마지막 학습 head와 serving-selected head를 가리킨다. `loss_weighting.json` 및 `report.loss_weighting`에 가중치와 train-only 정규화 근거를 기록한다.

`metrics.jsonl`의 epoch 1 이후 validation은 그 시점 학습 head의 결과다. `report.validation`과 `report.test`는 serving-selected head의 결과다. 이 둘을 같은 모델의 점수로 조용히 해석하지 않는다.

## 검색과 MLflow

Hyperopt v2 목적함수는 `1 - search_score`다. 실행기의 `child.search_validation`에 trained-only 점수를 보관하고, legacy `final_validation`은 serving-selected validation으로 유지한다. test는 검색·조건 선정에 사용하지 않는다.

기존 Child params 4개와 metrics 7개를 유지한다. loss 가중치는 run 이름·태그·설정 JSON에 표시한다. `jery.selected_epoch`, `jery.best_trained_epoch`, `jery.best_trained_score`, `jery.selection_outcome` 태그로 baseline 복원을 구분한다. 상세 지표는 JSON artifact에 저장한다.

2026-10-19 사용자 요청으로 추가한 공동 Hyperopt 탐색 plan은 `child_params`에 `window_none_loss_weight`를 명시해 필수 params를 5개로 확장한다. 기존 params 4개 plan은 그대로 지원한다. 새 plan에서는 LR과 함께 탐색한 가중치를 실제 Child param으로 기록하며, metrics 7개는 유지한다. 그 외 설정·진단은 JSON artifact다.

모델 artifact는 `model/`(serving), `model/best-trained/`, `model/last-trained/`로 구분한다. 원본 가중치·export·로그는 Git에 넣지 않는다. 완료된 checkpoint에서 후처리를 재개할 때 optimizer step을 추가하거나 test를 다시 평가하지 않는다.
