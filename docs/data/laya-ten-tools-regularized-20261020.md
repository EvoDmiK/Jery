# 10개 도구 정규화 재학습 — 2026-10-20

사용자의 파라미터 변경·재학습 요청에 따라 기존 도구·분할·base head 초기화를 유지하고 별도 Parent로 Hyperopt TPE 5회를 실행한다. 이전 Parent `c328674ecd9c4eabb1bf3d4dbc6c9dde`는 5개 Child와 artifact 업로드까지 완료됐다.

이전 train/validation 곡선에서 학습 정확도는 마지막 96.67~100%였지만 검증 최고점 이후 하락했다. 예를 들어 5번은 epoch 15 train 90%·validation 65%였고 epoch 25 train 100%·validation 55%, 마지막 validation 50%였다. 이를 정규화 실험의 근거로 사용한다. Test 점수로 범위나 후보를 선택하지 않는다.

| 설정 | 이전 | 이번 |
|---|---|---|
| Head LR (loguniform) | 1e-4~1e-3 | 1e-4~4e-4 |
| Weight decay | 0 | 0.01 / 0.05 / 0.1 탐색 |
| Label smoothing | 0 | 0.05 / 0.1 탐색 |
| Warmup optimizer steps | 0 | 10 |
| 최대 / 최소 관찰 epoch | 64 / 32 | 32 / 16 |
| Patience / min_delta | 5 / 0.004 | 6 / 0.004 |

고정 조건은 seed 3407, microbatch 4·누적 1, 선형 LR 감소, frozen FP16 encoder·FP32 routing head·BF16 autocast, train/validation/test 60/20/20, 동일한 10개 의미 선택지와 단일 창이다. 무효한 window_none 가중치는 1로 유지한다. 기존 실험의 학습 head를 이어 학습하지 않는다. 여러 설정을 함께 바꾸는 탐색이므로 개별 변수의 인과 효과는 주장하지 않는다.

MLflow Experiment Jery(47), Parent–Child, 필수 Child params 5개와 metrics 7개를 유지한다. 탐색된 weight_decay와 label_smoothing은 configuration·hyperopt JSON artifact에 저장한다. Child val 지표의 표 값은 마지막 epoch이며 선택된 head 검증 값은 report.json 및 Parent best_val_accuracy로 확인한다. test는 각 Child에서 선택된 head에 한 번 수행하고 탐색·선정에는 사용하지 않는다. 작은 미검토 합성 평가이므로 실제 일반화 성능의 근거가 아니다.

데이터 hermes-ten-tools-v1 / decision-export-v2, catalog hermes-ten-tools-b4648cc19a90, 기존 requirements-decision-cuda.txt를 사용한다. 초기 20개 암기 진단 통과 증거를 보존한다. 전용 모델 실행 설정을 configs/training에 두어 기존 모델 설정을 수정하지 않는다.

```sh
/home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-regularized-20261020.json \
  --model-config configs/training/laya-ten-tools-regularized-model-20261020.json \
  --state-dir outputs/laya-ten-tools-regularized-20261020/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/training/laya-ten-tools-regularized-model-20261020.json \
  --state-dir outputs/laya-ten-tools-regularized-20261020/tracking --detach
```

설정 계약 검사, 기존 데이터 검증(130개), CUDA 실험 계획 통합 테스트 4개를 통과했다. 제품 코드는 변경하지 않았다.

실제 Parent `75c5cfd07701439d88f29f0cf1158a46` / 첫 Child `6d703329748e4cdeb29108a652471c6f`로 실행했다. 첫 조건은 head_lr 0.00016114922909327275, weight_decay 0.05, label_smoothing 0.05다. epoch 9·optimizer step 135까지 실제 CUDA 학습과 유한 loss/gradient를 확인했다. CUDA 모델 테스트도 15개 통과했다(PYTHONPATH=src 설정). 이전 native readiness·독립 감사·암기 gate를 새 Parent에 업로드하고 SHA-256 read-back을 확인했다.
