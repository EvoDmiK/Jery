# Laya 전체 CUDA 학습과 Mac 동시 실행

Linux RTX 5070 Ti에서 기존 full-data 학습 계약을 적용한다. `laya_full` CLI가 config의 `backend`로 MLX/CUDA를 고른다. CUDA 구현은 `src/jery/model/laya_cuda.py`, 설정은 `configs/model/laya-cuda-20261019.json`이다.

## 실행 정책

- Mac Parent `9325d1ca84944fe281f94b519c59f305`의 진행 중 trial·설정·파일을 유지한다.
- 서버 검증 Parent: [8512d92300244b98a92ed07f0f64f4c2](https://mlflow.dove-nest.com/#/experiments/47/runs/8512d92300244b98a92ed07f0f64f4c2). 고정 학습률 1e-4, 한 에폭의 별도 Child로 준비 상태를 확인한다.
- CUDA Hyperopt Parent: [29552487a8a94b10ad942d9bbc331953](https://mlflow.dove-nest.com/#/experiments/47/runs/29552487a8a94b10ad942d9bbc331953). 처음에는 Mac 완료를 기다리도록 등록했으나, 사용자의 "CUDA도 바로 돌려줘" 지시에 따라 대기 supervisor만 종료하고 같은 Parent를 즉시 실행했다. Mac 학습은 유지한다. 대기 해제 기록은 Parent의 `control/immediate-start-override.json`에 보관했다.
- Hyperopt TPE 5회, 최대 64에폭/trial, 학습 데이터로만 40/40/10/10 oversampling, validation case accuracy로 선택, baseline epoch 0 포함, patience 5·min_delta 0.004를 유지한다.
- 각 Child에서 선택된 head의 test는 마지막에 한 번만 평가한다. 테스트 점수로 검색하거나 backend를 선택하지 않는다. 한 에폭 검증의 목적은 실행·시간·저장 확인이다.
- MLflow params 4개와 metrics 7개만 기록한다. 나머지는 JSON artifact, head는 safetensors로 보관한다. CUDA head 형식은 `unsloth-torch-laya-routing-v1`이며 MLX optimizer checkpoint와 호환된다고 가정하지 않는다.

## 서버 설정과 데이터

`/home/persona-ai/git/Jery-cuda`, `/home/persona-ai/envs/jery-cuda/bin/python`을 사용한다. 의존성은 `requirements-decision-cuda.txt`를 따른다. BF16 autocast, FP16 frozen encoder, FP32 routing head 14,770,945개, micro batch 4 / accumulation 1 / eval batch 4다. 프로세스의 CUDA allocator 한도는 GPU의 30%다. 이는 연산 점유율 제한이 아니다.

기존 IsaacLab·vision 프로세스를 중단하지 않는다. 다른 작업의 처리량 변화는 별도 측정하지 않았으며, 단순 생존 확인만으로 성능 영향이 없다고 결론 내리지 않는다.

합성 미검토 `mcp-routing-v2`, `decision-export-v1`, 카탈로그 `hermes-mcp-57e701d87790`를 사용한다. 원본 사례 668/248/246개가 exhaustive window 4,172/1,592/1,577개로 확장된다. 학습은 모든 원래 창을 유지한 균형 인덱스 8,530개, 2,133 optimizer step/epoch다. 도구 선택 전 state·questions만 특징으로 인코딩하며, native builder의 특징과 blind encoder가 일치하는지 전수 검증한다. 후보 밖 도구 선택은 공통 집계 평가에서 거부한다.

## 재현과 확인

저장소 루트에서 실행한다. 준비 스크립트는 기존 state가 있으면 같은 plan인지 확인하며, 원본 export·catalog와 설정을 JSON artifact로 올리고 재조회 체크섬을 확인한다.

```sh
PY=/home/persona-ai/envs/jery-cuda/bin/python
"$PY" scripts/prepare_laya_cuda_experiment.py \
  --state-dir outputs/laya-cuda-20261019/readiness --readiness
TORCHINDUCTOR_COMPILE_THREADS=4 "$PY" scripts/run_laya_full_experiment.py \
  --python "$PY" --model-config configs/model/laya-cuda-20261019.json \
  --state-dir outputs/laya-cuda-20261019/readiness --detach

# 준비 검증 완료 후 Mac과 독립적으로 즉시 실행한다.
"$PY" scripts/prepare_laya_cuda_experiment.py \
  --state-dir outputs/laya-cuda-20261019/hyperopt
TORCHINDUCTOR_COMPILE_THREADS=4 "$PY" scripts/run_laya_full_experiment.py \
  --python "$PY" --model-config configs/model/laya-cuda-20261019.json \
  --state-dir outputs/laya-cuda-20261019/hyperopt --detach
```

`outputs/laya-cuda-20261019/{readiness,hyperopt}/state.json`과 `supervisor.log`에서 대기·학습·완료·오류를 확인한다. 각 Child의 모든 step은 `outputs/model/laya-cuda-20261019/<local_id>/steps.jsonl`, 에폭 결과는 `metrics.jsonl`에 남긴다. baseline을 포함한 최적 head를 복원해 새 모델 인스턴스에 재로드하고, 전체 validation의 확률·선택과 encoder 해시를 비교한 뒤 test를 평가한다.

`--wait-for-run <MLflow run ID>`는 선택적으로 사용할 수 있다. 이 옵션을 주면 해당 run의 성공 완료를 기다리고 실패·중단 시 자동 시작하지 않는다. 현재 CUDA 실행에는 이 옵션을 사용하지 않는다.

중단된 CUDA Child는 같은 supervisor 명령으로 재개한다. 최신 native CUDA checkpoint의 모델·optimizer·scheduler·RNG를 HF Trainer로 복원한다. checkpoint 이후의 불완전 JSON 기록은 복구하고, checkpoint보다 뒤의 step·epoch 기록은 재실행한다. 이미 완성된 `report.json`이 있으면 재학습하지 않고 업로드를 재시도한다.

원본 checkpoint를 변경하지 않는 1-step 재개 확인:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src "$PY" scripts/check_laya_cuda_resume.py \
  --config outputs/model/laya-cuda-20261019/<local_id>/config.json \
  --checkpoint models/laya-cuda-20261019/<local_id>/trainer/checkpoint-2133 \
  --output outputs/laya-cuda-20261019/resume-check
```

이 확인은 별도 출력 경로에서 optimizer step 2,133→2,134 복원을 검증한다. 한 에폭 끝에 저장된 학습률은 0이므로 정확도 개선이나 추가 head 갱신을 입증하는 테스트는 아니다.

## 실제 검증 결과

Readiness Child `b4b307c8271f4e65beaa10e8f89df97f`가 전체 2,133 optimizer step을 완료했다.

| 항목 | 측정 시간 |
|---|---:|
| CUDA optimizer 계산 합계 | 243.61초 |
| CUDA 학습 호출: 한 에폭 + 검증 + native checkpoint | 271.85초 (4분 32초) |
| CUDA CLI 전체: 로드·인코딩·baseline·학습·평가·저장·새 인스턴스 복원·최종 test 포함 | 356.28초 (5분 56초) |
| Mac 이전 trial 첫 에폭의 학습·checkpoint·검증 | 1,149.71초 (19분 10초) |

Mac 기준은 기존 첫 Child `bb1c80ea642040878a392833ea2cadd1`의 첫 에폭 기록: `train_seconds=1024.6754`, validation `125.0370`의 합이다. 해당 구성에서 에폭 학습·검증은 약 4.2배 빨랐다. micro batch/누적 구성, backend precision과 학습률이 다르고 서버에 다른 GPU 작업이 있어 엄밀한 하드웨어 단독 비교는 아니다. 초기 32스텝의 0.1206초와 전체 에폭의 중앙 스텝 0.1158초가 비슷했다.

최대 PyTorch CUDA 할당은 1.359GiB, 예약은 1.393GiB다. 모든 split의 토큰 절단은 0이다. encoder 전체 파라미터 해시 불변, head 갱신과 새 인스턴스 복원 후 validation 확률 차이 0·동일한 case 선택을 확인했다.

정확도 개선은 없었다. baseline validation 16.13%에서 학습 1에폭 후 3.63%로 하락해 epoch 0을 선택·복원했다. 최종 validation 16.13%, test 15.04%는 복원한 초기 head의 결과다. 학습된 head는 native checkpoint에 남아 있고 배포용 최종 head와 구분된다.

공통·신규 테스트 112개, v1/v2 데이터 검증을 통과했다. 별도 출력에서 checkpoint를 재개해 trainer와 Adam optimizer의 step 2,133→2,134를 확인했고, 원본 checkpoint의 모든 파일 SHA-256은 유지됐다. 재개 결과는 readiness Parent의 `readiness/resume-check.json`에 있다.

완료된 한 에폭의 파일을 별도 경로에 복사한 후 `--resume`도 실제 실행했다. optimizer step을 추가하지 않고 post-training 복원·저장만 마쳤으며 기존 test 캐시를 재사용했다. 마지막 학습 head와 선택된 초기 head의 SHA-256이 각각 원본과 일치했다. 업로드·후처리 실패로 재개할 때 완료된 학습을 다시 수행하지 않는다.

MLflow에는 최적 serving head와 `model/last-trained/`의 실제 한 에폭 학습 head를 구분해 올렸다. readiness의 마지막 학습 head는 완료된 native checkpoint에서 추출하고 출처 해시를 기록했다. 다음 CUDA trial은 두 head를 자동으로 따로 보관한다. 최초 32스텝 결과는 [짧은 CUDA 실행 검증](laya-cuda-probe-20261019.md)에 있다.
