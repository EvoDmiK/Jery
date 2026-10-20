# Train600 참고 모델 기반 Hyperopt 20회 — 2026-10-20

사용자가 지정한 validation F1 실행과 seed3407·test accuracy >0.7·test macro F1 >0.7 실행의 실제 MLflow configuration.json/report.json을 확인해 탐색 범위를 정했다. 기존 train600 5회 Parent `88a5250cd54048d8a9593bab7e59d753`는 전부 완료된 상태에서 별도의 Parent 아래 20회를 순차 실행한다.

## 참고 실행

| Run ID | LR | WD | Smoothing | 최대 epoch / 선택 epoch | Test acc / F1 |
|---|---:|---:|---:|---:|---:|
| 90655e03d74f4ae5b77510eaf40e8033 | 0.00019706901574866268 | 0.05 | 0.05 | 16 / 11 | 0.60 / 0.54 |
| 1932f9d70c53484dbda1e41edacf1528 | 0.00035840220884888243 | 0.1 | 0.05 | 32 / 7 | 0.75 / 0.7067 |
| e62b5a7e334740e898fe3f4c4f5a284e | 0.00010059324102155978 | 0 | 0 | 64 / 26 | 0.75 / 0.73 |

세 실행 모두 seed3407·batch4다. 첫 실행의 표에 보이는 마지막 epoch14 validation F1은 0.8567(약0.86), 실제 선택 epoch11 validation accuracy/F1은 모두0.90이다. 나머지 두 실행만 test 양쪽 지표의 엄격한 >0.7 조건을 만족한다. 이들은 train60·이전 후보 설명으로 학습했고 첫 실행과 새 실험은 train600·v2.1 표현을 쓰므로 과거 성능을 직접 재현한다고 주장하지 않는다.

## 새 탐색 설정

- Hyperopt TPE 20회, startup6회. LR log-uniform 8e-5~5e-4, weight decay [0,0.05,0.1], label smoothing [0,0.05,0.1]. 세 참고 실행의 LR/WD/smoothing을 모두 포함한다.
- seed3407, micro batch4, grad accumulation1, warmup10 steps, linear scheduler, max sequence1024를 고정한다.
- 최대32epoch, 최소8epoch 관찰, patience6, min_delta0.004. 과거 선택 epoch26까지 수용할 예산을 두되 train600에서는 조기 종료할 수 있다. 매 epoch150 optimizer steps다.
- 기존 validation accuracy 기반 checkpoint·TPE 선택을 유지한다. Macro F1도 매 epoch 기록한다. 초기 head 보호와 best-trained 별도 보존을 유지한다.
- 각 Child의 최종 선택 head만 test20개로 한 번 평가한다. 현재 trial의 test를 TPE loss나 checkpoint 선택에 전달하지 않는다. 다만 사용자의 요청에 따라 과거 test 결과를 탐색 범위 설계에 사용했으므로 전체 실험은 test-informed 탐색이다. 반복 사용한 test 성능을 독립 최종 일반화 평가로 해석하지 않는다.

모든 trial은 동일한 pinned Laya base에서 새로 시작하고 frozen FP16 encoder·FP32 head/BF16 autocast를 사용한다. 데이터는 hermes-ten-tools-v3 train600/validation20/test20, catalog hermes-ten-tools-2289142ab57a, 표현 catalog-grounded-criteria-v2.1-native-budget다. 합성·미검토 정답에 대한 탐색 실험이다. 기존 모델 SHA, CUDA 메모리 비율0.3과 다른 GPU 작업을 보존한다.

필수 params5개·metrics7개만 tracking하고 전체 설정·참고 실행·데이터·epoch JSON·학습 head/base 참조는 Artifacts로 남긴다. MLflow 표의 val은 마지막 epoch이며 선택 checkpoint의 지표는 details/report.json에서 확인한다.

## 재현

설정: `configs/training/laya-ten-tools-train600-search20-20261020.json`, `configs/training/laya-ten-tools-train600-search20-model-20261020.json`.

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-train600-search20-20261020.json \
  --model-config configs/training/laya-ten-tools-train600-search20-model-20261020.json \
  --state-dir outputs/laya-ten-tools-train600-search20-20261020/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/training/laya-ten-tools-train600-search20-model-20261020.json \
  --state-dir outputs/laya-ten-tools-train600-search20-20261020/tracking --detach
```

기존 state/lock/process를 확인하고 supervisor를 중복 실행하지 않는다. 상태와 완료 여부는 서버 tracking/state.json 및 MLflow를 함께 확인한다. 데이터 검증 PASS(130 examples/10 tools/65 isolated scenarios), CUDA 계획 통합 테스트5개 PASS, 새 plan/model 및 참고값 범위 검증 PASS. 실행기 코드는 변경하지 않았고 추가 의존성도 없다.
