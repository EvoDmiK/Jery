# 10개 도구 배치·시드 10회 비교

사용자가 배치 사이즈·시드를 바꿔 10회 테스트하도록 요청했다. 이전 정규화 탐색 Parent `75c5cfd07701439d88f29f0cf1158a46`는 완료됐다. 검증으로 선정된 4번 Child `ca0d15030d6543a99ba6a7084679d6bc`의 LR 0.0003538647078409709, weight_decay 0.05, label_smoothing 0.05를 고정한다. 후보 선택에 test를 사용하지 않는다.

이전 Hyperopt로 찾은 설정에서 **배치 4·8 × 학습 시드 3407~3411**의 완전 교차 10회를 실행한다. 이번 단계는 적응형 TPE 대신 사전 고정한 반복 비교다. 같은 시드에서 배치 두 개를 연속 실행하고 모든 모델은 같은 원 base에서 새로 시작한다. 인코더와 pretrained head 초기 가중치는 동일하며 시드는 학습 shuffle/dropout 등 난수에 적용된다. 데이터 분할·선택지 순서·정답은 재생성하지 않는다.

고정 조건: train/validation/test 60/20/20, hermes-ten-tools-v1 / decision-export-v2, catalog hermes-ten-tools-b4648cc19a90, grad_accum 1, 최대 32에폭·최소 16에폭·patience 6·min_delta .004, warmup 10 optimizer step·linear scheduler, frozen FP16 encoder·FP32 head·BF16 autocast. Effective batch는 4 또는 8이며 각각 epoch당 optimizer step은 15 또는 8이다. 같은 epoch 수가 같은 update 수를 의미하지 않으며 실제 steps·종료 epoch·시간을 JSON에 보존한다.

MLflow Jery(47) 새 Parent 아래 Child 10개로 기록한다. 필수 params 5개·metrics 7개는 유지한다. 실제 trial seed와 effective_batch_size를 실행 override에서 계산해 기록하도록 공통 runner를 수정했다. 기존 fixed trial 형식은 유지하면서 seed/micro_batch/grad_accum만 선택적으로 허용하고 정수·범위를 검사한다. 조건과 전체 설정·epoch 이력은 JSON artifact다. Child 표의 val 값은 마지막 epoch이며 선정 head 검증은 report.json 및 Parent 지표로 확인한다.

반복 검증 점수로 선정하고 test는 최종 선택 head에 대해 기록한다. 20개 검증·평가 사례는 작고 합성 미검토 상태이며 반복 관찰된 평가이므로 실제 성능 향상으로 단정하지 않는다.

설정과 원본 데이터 검증, tracking 테스트 11개·CUDA 계획 테스트 5개를 통과했다. Trial별 override가 CLI 학습 입력과 MLflow seed·batch에 동일하게 반영되는지, 기본값 보존·잘못된 override 거부를 검사했다.

```sh
/home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-batch-seed-20261020.json \
  --model-config configs/training/laya-ten-tools-batch-seed-model-20261020.json \
  --state-dir outputs/laya-ten-tools-batch-seed-20261020/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/training/laya-ten-tools-batch-seed-model-20261020.json \
  --state-dir outputs/laya-ten-tools-batch-seed-20261020/tracking --detach
```
