# Laya CUDA 서버 준비와 짧은 학습 검증

2026-10-19 `persona-ai@persona-ai`에 Tailscale SSH 접속 후 별도 checkout과 가상환경을 준비했다.

- 코드: `/home/persona-ai/git/Jery-cuda`, 브랜치 `dev-codex`
- 환경: `/home/persona-ai/envs/jery-cuda`, Ubuntu 24.04.4 / Python 3.12.3
- 장치: RTX 5070 Ti 16GB, NVIDIA 드라이버 580.178.04, PyTorch 2.11.0+cu128
- 기존 IsaacLab·vision 작업은 유지했다. 벤치마크 전에도 GPU 약 7.2GiB와 연산 자원을 사용 중이었다.
- Mac M4의 기존 Hyperopt Parent `9325d1ca84944fe281f94b519c59f305`는 계속 실행한다. 이 검증으로 trial을 옮기거나 이어 학습하지 않았다.

## 실행 방법

별도 환경에서 설치한다. 드라이버나 기존 환경을 변경할 필요는 없다.

```sh
python3 -m venv /home/persona-ai/envs/jery-cuda
/home/persona-ai/envs/jery-cuda/bin/python -m pip install -r requirements-decision-cuda.txt
/home/persona-ai/envs/jery-cuda/bin/python -m pip check

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src \
  /home/persona-ai/envs/jery-cuda/bin/python scripts/probe_laya_cuda.py \
  --output outputs/laya-cuda-probe-20261019-final-b1 --steps 32

HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src \
  /home/persona-ai/envs/jery-cuda/bin/python scripts/probe_laya_cuda.py \
  --output outputs/laya-cuda-probe-20261019-final-b4 --steps 32 --micro-batch 4 --accumulation 1
```

모델과 export는 Git 밖에 둔다. 기존 Mac의 `models/laya-multilingual/`과 `outputs/laya-full-20261017/export/`를 같은 상대 경로에 복사했다. 모델 SHA-256은 `9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`, revision은 `1720e3e3357cfe1e281542e223f8273b0890ca34`다. export는 manifest 체크섬을 검증한다.

## 실제 결과

Unsloth `DecisionTrainer`의 PyTorch CUDA 경로로 각각 32 optimizer step을 실행했다. 처음 5 step을 제외한 중앙값이다.

| micro batch × accumulation | 유효 배치 | 중앙 스텝 시간 | 예제/초 | PyTorch 최대 할당 |
|---|---:|---:|---:|---:|
| 1 × 4 | 4 | 0.1461초 | 27.37 | 0.965GiB |
| 4 × 1 | 4 | 0.1206초 | 33.18 | 1.138GiB |

학습 호출 전체 시간은 각각 5.21초와 4.37초다. 이 시간은 모델 로드·데이터 인코딩·추론 준비를 포함하지 않는다. 최대 할당은 해당 Python 프로세스의 PyTorch 텐서 메모리이며 전체 GPU 사용량과 다르다. 프로세스 할당 한도는 GPU의 30%로 설정했다.

Mac에서 앞서 관측한 약 0.47초/스텝보다 이번 짧은 실행이 빨랐다. CUDA는 BF16 autocast, Mac은 MLX이며, 동시 GPU 작업·배치 구성·샘플 순서도 달라 엄밀한 하드웨어 단독 비교는 아니다. 전체 epoch 및 평가 처리량이나 최종 정확도는 이번에 측정하지 않았다.

- 전체 train 4,172개 window의 인코딩 검증: 누락·절단 0, 최대 950토큰.
- 기존 40/40/10/10 균형 샘플링 유지: 8,530개 인덱스로 dataset 구성. 실제 각 검증에서 소비한 양은 128개 예제이며 전체 epoch를 수행하지 않았다.
- encoder는 고정하고 `head`, `scorer`, `type_emb`의 14,770,945개 파라미터만 학습한다. native 모델의 별도 `act_head`는 고정해 MLX 학습 범위와 맞췄다.
- FP16 frozen encoder / FP32 head / BF16 autocast 사용. 초기 FP16 autocast 시도에서 gradient overflow가 보여 그 시도는 비교 결과에서 제외했다.
- 최종 두 실행 모두 유한 loss·gradient, head 갱신, encoder 파라미터 표본 불변, head 저장·재로드 값 일치, 재로드 후 logits 일치를 확인했다.
- `pip check`: 의존성 충돌 없음. 공통 Laya 테스트 11개, v1 130개·v2 1,162개 데이터 검증 통과.

원본 결과는 서버 `outputs/laya-cuda-probe-20261019-final-b{1,4}/report.json`과 `head.safetensors`, 로컬 수집본은 `outputs/laya-cuda-probe-20261019/`에 둔다. 전체 의존성은 `pip-freeze.txt`로 기록한다. 상세 정보는 JSON artifact에 보관하며 기존 학습 trial의 정확도 메트릭에 섞지 않는다.

## 남은 작업

이 스크립트는 CUDA 실행·속도 검증용이다. 기존 전체 학습기는 MLX 경로이므로 Linux에서 그대로 실행할 수 없다. 전체 CUDA 학습 전에는 공통의 window 집계·검증 콜백·Hyperopt·MLflow Parent/Child 정책을 연결하고, 새 backend의 baseline 평가와 checkpoint 재개를 검증해야 한다. Mac optimizer checkpoint를 CUDA에서 그대로 재개할 수 있다고 가정하지 않는다.

합성 미검토 데이터 `mcp-routing-v2`, export 계약 `decision-export-v1`, 도구 카탈로그 `hermes-mcp-57e701d87790`를 사용했다. 실제 요청에 대한 성능 개선을 입증한 실험은 아니다.
