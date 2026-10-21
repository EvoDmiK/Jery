# 다음20회 LR1e-3~5e-3 — 2026-10-21

사용자 요청으로 다음 Hyperopt20회의 head LR을 log-uniform1e-3~5e-3로 지정했다. 기존 상한1e-3 Parent6261a1bdca194867a835cbfb2216493f는 확인 시13개 완료/14번째 실행 중이어서 설정을 변경하지 않고, 성공 완료 뒤 새20회가 자동 시작하도록 기존 wait-for-run 경로를 재사용한다.

다른 조건은 TPE startup6, WD[0,0.05,0.1], smoothing[0,0.1], seed3407, batch4/누적1, 최대32epoch/최소8/patience6/min_delta0.004, warmup10 optimizer steps, linear scheduler, max_grad_norm1.0을 유지한다. 매 trial은 고정 base에서 새로 시작한다. CUDA frozen encoder·학습 head/scorer/type_emb 경로를 유지한다. 기존 참고6개 실행의 LR은 새 범위 밖이며 이전 정규화 범위의 출처로만 남긴다.

데이터 hermes-ten-tools-v3 train600/validation20/test20, catalog hermes-ten-tools-2289142ab57a/v2.1 표현과 base SHA, memory fraction0.3을 유지한다. 미검토 합성 정답·test-informed 범위 설계의 탐색 실험이다. TPE와 checkpoint 선택은 validation accuracy만 사용하고 최종 선택 head test는 각 Child에 한 번 수행한다. Params5/metrics7 외 상세 JSON·데이터·head/base Artifact 정책도 동일하다.

## 추가로 조정 가능한 값

아래 범위는 다음 비교용 제안이다. 이번20회에는 적용하지 않았다. 실제 CUDA 호출은 src/jery/model/laya_cuda.py의 TrainingArguments와 head freeze/optimizer 경로에서 확인했다.

| 항목 | 현재 | 다음 비교 후보 | 목적 |
|---|---|---|---|
| warmup_steps | 10 | 75,150 | 높은 peak LR에 도달하는 초기 속도 비교; batch4에서150steps/epoch |
| effective batch | 4 | 4,8,16 | micro_batch 또는 grad_accum으로 update당 사례 수 비교 |
| max_grad_norm | 1.0 | 0.5,1.0,2.0 | gradient clipping 강도 비교 |
| lr_scheduler | linear | linear,cosine | peak 이후 LR 감소 곡선 비교 |
| weight_decay / label_smoothing | 기존 TPE 탐색 | 우선 기존 범위 유지 | 현재도 탐색 중인 정규화 값 |

개별 모델 설정은 위 값을 받지만 현재 Hyperopt 자동 공간은 LR/WD/smoothing만 지원한다. Warmup/clipping/scheduler를 TPE로 함께 탐색하려면 runner의 override·검증 범위를 확장해야 한다. Batch/seed는 기존 fixed-trial override로 비교할 수 있다. Encoder는 고정되어 CUDA 경로의 encoder_learning_rate를 조절해도 encoder 학습은 생기지 않는다. 제안 범위는 프로젝트 조건에 대한 판단이며 성능 개선을 보장하는 문헌값이 아니다.

공식 근거: [Hugging Face scheduler/warmup](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules), [TrainingArguments의 clipping/accumulation](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py), [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW). 높은 LR 비교 뒤 warmup과 batch를 먼저 비교하는 것을 제안한다.

## 재현과 검증

설정은 configs/training/laya-ten-tools-train600-lr1to5e3-20261021.json 및 laya-ten-tools-train600-lr1to5e3-model-20261021.json이다.

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
 --plan configs/training/laya-ten-tools-train600-lr1to5e3-20261021.json \
 --model-config configs/training/laya-ten-tools-train600-lr1to5e3-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr1to5e3-20261021/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
 --python /home/persona-ai/envs/jery-cuda/bin/python \
 --model-config configs/training/laya-ten-tools-train600-lr1to5e3-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr1to5e3-20261021/tracking \
 --wait-for-run 6261a1bdca194867a835cbfb2216493f --detach
```

실행기 코드/추가 의존성 변경 없음. 데이터 검증 PASS(130examples/10tools/65isolated scenarios), 새 plan/model 계약 및 LR 범위/20회/기타 조건 보존 assert PASS. 기존 CUDA 계획 테스트5개도 확인한다. 기존 supervisor 중복 실행을 피하고 upstream FINISHED 때만 다음 Child를 생성한다. 대기 등록과 실제 학습 시작/완료는 구분한다.
