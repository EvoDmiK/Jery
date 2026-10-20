# Train600 F1 조건 기반 범위 축소 — 2026-10-21

이전 Parent `dca0889273a641eaae6e4a92bfa9efc9`의 20개 Child가 모두 FINISHED이고 서버 state도 finished/error null임을 확인했다. MLflow 표의 val_macro_f1 >=0.75 및 test_macro_f1 >=0.70을 동시에 만족한 6개 실행의 실제 details/report.json 설정을 기준으로 새 20회 탐색을 구성한다. 모든 참고 실행은 선택 checkpoint의 validation F1도0.75 이상이었다. 표 val은 마지막 epoch, test는 최종 선택 head이므로 선택 head의 val도 따로 대조했다.

| Run | LR | WD | Smoothing | Val F1 마지막 / 선택 head | Test F1 |
|---|---:|---:|---:|---:|---:|
| 87e3fbaf6f3848b898e02a3ecf089551 | 0.0001503249 | 0.05 | 0.0 | 0.7933 / 0.84 | 0.7433 |
| 62b290a1f0b64728838865fd265ad30d | 0.0001507529 | 0.05 | 0.1 | 0.7733 / 0.76 | 0.7033 |
| a921dc11102849a6becf9d59eed28738 | 0.0002223984 | 0.1 | 0.1 | 0.8 / 0.8933 | 0.7933 |
| 39fe7d47d62d4e25bdcd21c9e6124e86 | 0.0003068819 | 0.0 | 0.0 | 0.8433 / 0.8933 | 0.72 |
| 2df2593adf2a4605978ba58c03fb313d | 0.0003878264 | 0.1 | 0.1 | 0.8433 / 0.9467 | 0.7933 |
| c79ecdc6091a46b98e938d5817d2743b | 0.0004063897 | 0.1 | 0.1 | 0.78 / 0.9467 | 0.74 |

## 변경과 유지 조건

LR log-uniform 8e-5~5e-4를 **1.4e-4~4.2e-4**로 좁히고 smoothing [0,0.05,0.1]을 **[0,0.1]**로 줄였다. WD는 참고 실행에서0/0.05/0.1 모두 관찰되어 세 값을 유지한다. 모든 참고값을 새 범위에 포함하는지 검증했다.

기존 실행기를 재사용해 Hyperopt TPE20회/startup6, seed3407, batch4/누적1, 최대32epoch/최소8/patience6/min_delta0.004, warmup10와 linear scheduler를 유지한다. 매 trial은 pinned Laya base에서 새로 시작한다. Frozen FP16 encoder·FP32 head/BF16 autocast, CUDA memory fraction0.3, 기존 SHA 및 길이 예산을 유지하고 다른 GPU 작업을 변경하지 않는다.

데이터 hermes-ten-tools-v3 train600/validation20/test20, catalog hermes-ten-tools-2289142ab57a, 표현 catalog-grounded-criteria-v2.1-native-budget. 미검토 합성 정답의 탐색 실험이다. 과거 test를 범위 설정에 사용한 test-informed 탐색이며 반복 test를 독립 일반화 평가로 해석하지 않는다. TPE loss와 checkpoint 선택은 기존 validation accuracy만 사용하고 각 Child의 최종 선택 head test는 한 번 수행한다. Macro F1은 매 epoch 기록한다. 필수 params5·metrics7 외 전체 설정·참고 결과·데이터·epoch 이력과 head/base는 JSON/model Artifacts로 보관한다.

## 재현과 검증

설정은 `configs/training/laya-ten-tools-train600-refine20-20261021.json`, `configs/training/laya-ten-tools-train600-refine20-model-20261021.json`이다. 추가 의존성과 실행기 코드 변경은 없다.

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-train600-refine20-20261021.json \
  --model-config configs/training/laya-ten-tools-train600-refine20-model-20261021.json \
  --state-dir outputs/laya-ten-tools-train600-refine20-20261021/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/training/laya-ten-tools-train600-refine20-model-20261021.json \
  --state-dir outputs/laya-ten-tools-train600-refine20-20261021/tracking --detach
```

데이터 검증 PASS(130 examples/10tools/65 isolated scenarios), CUDA 계획 통합 테스트5개 PASS, 새 plan/model 계약과 참고6개 threshold·선택 head val·범위 포함 검증 PASS. 학습 상태는 서버 state.json과 MLflow를 따른다. Supervisor 중복 실행 전 기존 process/lock/state를 확인한다.
