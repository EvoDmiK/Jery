# 고정 10개 도구 train600 CUDA 학습 — 2026-10-20

사용자 요청으로 준비한 hermes-ten-tools-v3의 train600/validation20/test20 전부를 사용하는 CUDA Hyperopt 학습을 시작했다. MLflow Jery(47)의 Parent는 `88a5250cd54048d8a9593bab7e59d753`, 첫 Child는 `4a01c09c5a2c4edaa2b341d415901a39`다. 다섯 Child를 이 Parent 아래 순차 실행한다. 이 문서는 시작 시점 기록이며 완료 결과는 MLflow와 서버 state.json을 따른다.

## 설정과 선택 정책

기존 학습 실행기를 재사용한다. 설정은 `configs/training/laya-ten-tools-train600-20261020.json`과 `laya-ten-tools-train600-model-20261020.json`, 서버 코드 기준 커밋은 ce638ba다. 매 Child는 같은 pinned Laya base(1720e3e3357cfe1e281542e223f8273b0890ca34, weight SHA 9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204)에서 새로 시작한다. 이전 학습 head를 이어 학습하지 않는다. FP16 encoder를 고정하고 FP32 head를 BF16 autocast로 학습한다.

- Hyperopt TPE 5회, startup 2회. head LR log-uniform 1e-5~2e-4만 탐색한다.
- Weight decay 0.05, label smoothing 0.05, seed3407, batch4/누적1, warmup10 optimizer steps와 linear scheduler를 고정한다.
- 최대16에폭, 최소4에폭 관찰, patience3/min_delta0.004. 데이터 증가로 epoch당 optimizer step은150이다.
- 이전 검증 곡선과 증가한 update 수를 고려해 낮은 LR 범위를 탐색한다. 데이터·학습 설정이 함께 바뀌는 탐색이므로 순수 데이터 보강 효과의 인과 비교로 주장하지 않는다.
- 최고 trained validation 정확도로 Hyperopt를 진행한다. Serving은 초기 head보다 엄격히 좋아질 때만 trained head를 선택한다. best-trained head는 baseline fallback과 별개로 보존한다.
- Test는 각 Child의 최종 선택 head에 한 번만 수행하고 탐색·선정에 사용하지 않는다. 같은 합성 heldout을 반복 관찰한 회귀 결과이며 실사용 정확도를 보증하지 않는다.

필수 Child params5개·metrics7개 계약을 유지하고 전체 설정, 에폭 이력·상세 지표와 데이터는 JSON Artifacts에 저장한다. MLflow 표의 val은 마지막 에폭 값이며 최종 선택 head의 검증은 report.json 및 selected_epoch 태그를 확인한다. 완료 시 head와 base 참조를 업로드하고 SHA를 read-back 검증하는 기존 supervisor를 사용한다.

## 실행과 실제 확인

서버 반영 후 Parent 초기 데이터 업로드 중 회사 PC의 인터넷이 끊겼다. Tailscale offline 및 SSH timeout을 확인했으며 이때 Child 학습은 시작되지 않았다. 사용자 재연결 확인 후 기존 준비 프로세스의 timeout 종료와 미시작 상태를 확인하고 같은 Parent/state로 prepare를 재실행했다. Parent를 중복 생성하지 않았다.

Supervisor PID208808을 detached session으로 실행했다. 첫 Child LR은 1.4392629715379414e-5다. 실제 CUDA optimizer step1050과 6개 에폭의 MLflow 동기화를 확인했다. 초기 validation60%, epoch1~5는55%, epoch6은60%였다. 이 시점에는 baseline을 넘지 않았다. peak allocated GPU memory는 약1.085GB였다. 서버의 다른 GPU 작업은 중단하거나 변경하지 않았다.

서버 기준 export는 `outputs/laya-ten-tools-training-v3-final-20261020/augmented/`, 추적 상태는 `outputs/laya-ten-tools-train600-20261020/tracking/state.json`, supervisor.log 및 trial별 모델·출력은 설정 파일의 models_dir/outputs_dir에 있다. SSH 접속과 분리된 supervisor가 나머지 trials·지표 동기화·모델 업로드를 이어간다. 학습 중 tracking 연결이 끊기면 로컬 학습을 보존하고 연결을 재시도한다. PC 자체의 전원·프로세스 종료는 별도 상태 확인이 필요하다.

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-train600-20261020.json \
  --model-config configs/training/laya-ten-tools-train600-model-20261020.json \
  --state-dir outputs/laya-ten-tools-train600-20261020/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/training/laya-ten-tools-train600-model-20261020.json \
  --state-dir outputs/laya-ten-tools-train600-20261020/tracking --detach
```

기존 supervisor가 살아 있으면 위 detach를 중복 실행하지 않는다. state·lock·process와 MLflow를 함께 확인한다. 이번 변경은 실행 설정·문서뿐이며 새 의존성은 없다. 원본 데이터 검증 PASS(130 examples/10 tools/65 isolated scenarios), CUDA 계획 통합 테스트5개 PASS, 새 설정의 validate_explicit_plan PASS였다. 데이터 v3의 최종 native audit은 600/20/20에서 skipped/truncated/feature mismatch/target mismatch0이었다. 가중치와 대형 실험 결과는 Git에서 제외한다.
