# Laya CUDA 학습 정책 `laya-routing-v2` (2026-10-19)

선택형 CUDA 학습 정책이다. `training_policy_version`이 없으면 기존 MLX·CUDA 동작을 그대로 유지한다. MLX backend에서 이 키를 주면 거부한다. 이번 작업은 구현과 단위 검증만 했고, 학습은 실행하지 않았다.

## 배경 (관측된 붕괴)

- validation 248건 중 232건이 `select_tools`인데, 1,592개 창 중 1,305개는 정답이 `window_none`인 음성 창이다.
- Mac trial 1: 양성 창 정확도 26.81% → 1.70%, 전체 창 정확도 8.79% → 78.96%, case 정확도 16.13% → 4.84%. 이전 실행은 모두 epoch 0을 선택했다.
- 기존 정책은 baseline epoch 0과 학습된 epoch를 같은 patience 기준으로 비교한다. 그래서 학습이 baseline 아래에서 회복하기 전에 멈출 수 있었고, 학습된 최고 checkpoint를 따로 남기지 않았다.

## 변경

설정(`configs/model/laya-cuda-routing-v2-20261019.json`): `training_policy_version: "laya-routing-v2"`, `window_none_loss_weight`((0, 1], 기본 1.0), `validation_callback.{patience 5, min_delta .004, min_epochs 10}`, `label_smoothing 0`, 최대 64 epoch. 통제 진단은 coordinator가 `epochs 10`과 weight 1.0/0.5/0.25로 override한다.

- **데이터**: 원본 train 창을 모두 유지하고 기존 4:4:1:1 oversampling을 그대로 쓴다. 정답이 `window_none`인 sampled 창만 가중치 `w`를 받고 나머지는 1이다.
- **손실**: 창마다 native soft cross entropy(같은 label smoothing 식)를 계산하고 `w_i / mean_train(w)`를 곱한 뒤 microbatch 평균을 낸다. 정규화 상수는 sampled train 전체에서 한 번 계산한다. minibatch 합으로 나누지 않으므로 microbatch 1에서도 상대 가중치가 상쇄되지 않는다. 가중치가 모두 1이면 native 손실과 같다.
- **입력 차단**: 가중치는 collator가 `loss_weight` 열로만 붙인다. `target`과 함께 forward 전에 제거한다. 모델이 받는 키는 `input_ids, attention_mask, marker_pos, marker_mask, qtype`뿐이다. encoder·action head 동결, routing 파라미터 수(14,770,945)와 feature-blind 검사는 그대로 둔다.
- **선택 정책(`RoutingPolicy`)**:
  - best trained는 epoch ≥ 1 중 validation case_accuracy의 정확한 최댓값이다. 동점이면 앞 epoch를 남긴다. min_delta와 무관하게 `best-epoch-{e}`로 저장한다.
  - serving은 best trained가 baseline보다 엄격히 높을 때만 그 epoch를 고르고, 아니면 epoch 0을 고른다.
  - patience는 학습된 epoch끼리만 min_delta로 비교한다. bad epoch는 `epoch > min_epochs`부터 센다.
  - resume은 durable epoch history를 그대로 replay한다. 정지했거나 최대 epoch에 도달한 checkpoint는 다시 불러오기만 하고 step을 추가하지 않는다(`resume_adds_steps`).

## 산출 JSON

- `report.selection`: 기존 `best_epoch`/`best_score`(serving 선택), `selected_epoch`, `completed_epochs`, `max_epochs`, `reason`에 다음을 더한다: `policy_version`, `best_trained_epoch`, `best_trained_score`, `search_score`(= best_trained_score), `trained_bad_epochs`, `min_epochs`, `baseline_score`, `patience_reference_score`. `bad_epochs`는 trained_bad_epochs와 같다.
- `report.best_trained`: `{artifact, artifact_sha256, epoch, validation_case_accuracy, format, base_model_safetensors_sha256, loader}`. `report.last_trained`, `final/`, `report.restore`는 기존과 같다.
- `loss_weighting.json`과 `report.loss_weighting`: `original_counts`, `sampled_counts`, `sampled_weight_mass`, `train_mean_weight`, `normalization`.

## 검증 (실제 결과)

```sh
PYTHONPATH=src tmp/venv-decision/bin/python -m unittest tests.model.test_laya_cuda tests.model.test_laya_full tests.model.test_laya_pilot
# Ran 19 tests ... OK
```

- weight 1 손실이 native `_decision_loss`와 같다(smoothing 0, 0.1). 0.5/0.5 soft target 값도 확인했다.
- microbatch 1에서 weight .25의 head gradient가 weight 1의 정확히 1/4이다. forward 입력 키에 target·weight가 없고 encoder gradient는 None이다.
- 정책: baseline 아래에서 회복해도 조기 정지하지 않는다. best trained와 serving baseline fallback을 따로 고르고, min_epochs 전 bad epoch는 세지 않는다. replay가 정확하고, 정지·최종 checkpoint는 step을 추가하지 않는다.
- 별도 smoke로 fake model에 HF `Trainer.train`(CPU, microbatch 1)을 돌렸다. weight .25 step의 손실이 weight 1 step의 약 1/4이었고 encoder는 바뀌지 않았다. 실제 CUDA 학습은 실행하지 않았다.

## 한계

- 미검토 합성 데이터 위의 탐색 실험용이다. 실제 사용 정확도를 주장하지 않는다.
- 첫 CUDA 실행에서 `loss_weighting.json`과 `report.best_trained`가 실제로 생성되는지 확인해야 한다.
