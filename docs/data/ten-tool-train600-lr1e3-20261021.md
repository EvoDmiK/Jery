# 다음20회 LR 상한1e-3 — 2026-10-21

사용자가 다음 탐색 LR 상한을1e-3로 지정했다. 기존 refine20 Parent `6e2c92b081534cc8a8e3b1271e3c0e9c`의 실행 중 조건을 변경하지 않고, 완료 뒤 자동 시작할 별도 Parent–Child20회 계획을 준비했다. 준비 당시 기존20회 중2개 완료/3번째 실행 중이었다.

유일한 학습 조건 변경은 head LR log-uniform 범위의 상한4.2e-4→1e-3다. 하한1.4e-4, Hyperopt TPE20/startup6, WD[0,0.05,0.1], smoothing[0,0.1], seed3407, batch4/누적1, 최대32epoch/최소8/patience6/min_delta0.004, warmup10/linear를 유지한다. 고정 base에서 매 trial을 새로 시작한다.

데이터 hermes-ten-tools-v3 train600/validation20/test20, catalog hermes-ten-tools-2289142ab57a/v2.1 표현과 base SHA를 그대로 사용한다. Frozen encoder/head 학습·CUDA memory fraction0.3과 기존 다른 GPU 작업을 보존한다. 참고6개 실행과 test-informed 범위 설계의 탐색 한계도 유지한다. TPE/checkpoint 선택은 validation accuracy만 사용한다. Test는 최종 선택 head에 한 번만 수행한다. 필수 params5·metrics7, 상세 JSON/모델 Artifact 정책은 동일하다.

## 실행

설정: `configs/training/laya-ten-tools-train600-lr1e3-20261021.json`, `configs/training/laya-ten-tools-train600-lr1e3-model-20261021.json`.

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
 --plan configs/training/laya-ten-tools-train600-lr1e3-20261021.json \
 --model-config configs/training/laya-ten-tools-train600-lr1e3-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr1e3-20261021/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
 --python /home/persona-ai/envs/jery-cuda/bin/python \
 --model-config configs/training/laya-ten-tools-train600-lr1e3-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr1e3-20261021/tracking \
 --wait-for-run 6e2c92b081534cc8a8e3b1271e3c0e9c --detach
```

기존 runner의 wait-for-run은 upstream MLflow Parent가 FINISHED일 때만 다음 학습을 시작한다. RUNNING이면15초 간격으로 대기하고 FAILED/KILLED면 후속 학습을 시작하지 않고 오류를 남긴다. 대기 중 새 Child/GPU 학습은 없다. 중복 supervisor를 실행하지 않는다.

데이터 검증 PASS(130 examples/10tools/65 isolated scenarios), CUDA 계획 통합 테스트5개 PASS, 새 plan 계약/LR 상한 및 나머지 조건 보존 검증 PASS. 실행기 변경·추가 의존성 없다. 대기 등록과 학습 시작·완료는 다른 상태이며 실제 상태는 서버 state/MLflow를 따른다.

## 등록 상태

[다음 Parent 6261a1bdca194867a835cbfb2216493f](https://mlflow.dove-nest.com/#/experiments/47/runs/6261a1bdca194867a835cbfb2216493f)를 생성하고 데이터·설정 Artifact SHA read-back을 검증했다. 실행 코드/설정 기준73c3db9를 서버에 fast-forward 반영했다. Detached runner에 --wait-for-run으로 기존 refine20 Parent를 지정했다.
