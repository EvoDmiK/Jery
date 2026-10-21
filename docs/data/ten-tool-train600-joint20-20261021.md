# Train600 다중 파라미터 Hyperopt20회 — 2026-10-21

사용자가 LR만 바꾸는 실험에 더해 warmup·유효 배치·gradient clipping·scheduler도 함께 적용하도록 요청했다. LR-only 대기 Parent `8e27ea4395af45c9ba1fc429e01dab44`에 Child가0이고 waiting_for_upstream임을 확인한 뒤 해당 supervisor225880만 SIGTERM으로 중단했다. 서버 state interrupted, MLflow KILLED와 Child0을 확인했다. 실제 학습 중인 이전 Parent `6261a1bdca194867a835cbfb2216493f`와 다른 GPU 작업은 유지한다. 그 Parent 성공 완료 뒤 새 공동 탐색20회를 시작한다.

## 실제 공동 탐색 공간

| 항목 | 후보 |
|---|---|
| Head LR | log-uniform1e-3~5e-3 |
| Weight decay | 0,0.05,0.1 |
| Label smoothing | 0,0.1 |
| Warmup optimizer steps | 10,75,150 |
| Gradient accumulation | 1,2,4 |
| Effective batch | micro batch4 × accumulation =4,8,16 |
| Max gradient norm | 0.5,1,2 |
| LR scheduler | linear,cosine |

Hyperopt TPE20회/startup6으로 위7개 변수를 함께 탐색한다. Warmup10과 다른 기존 값도 포함해 baseline 근방을 비교할 수 있게 했다. 20회는7차원 공간의 거친 탐색이며 전 조합 비교나 개별 변수의 인과 효과 검증이 아니다. Micro batch4를 고정해 GPU activation 메모리 사용을 유지하고 accumulation으로 유효 배치를 늘린다. Batch가 달라지면 epoch당 optimizer step과 전체 scheduler 길이도 달라진다. Warmup은 epoch 비율이 아닌 optimizer step 수다.

Seed3407·최대32epoch·최소8·patience6/min_delta0.004, 데이터 hermes-ten-tools-v3 train600/validation20/test20, catalog hermes-ten-tools-2289142ab57a/v2.1 표현, base revision/SHA를 유지한다. 매 trial은 고정 base에서 새로 시작한다. CUDA encoder는 고정하고 head/scorer/type_emb를 학습한다. Native max_len1024/head256, memory fraction0.3과 기존 다른 작업을 유지한다.

TPE/checkpoint 선택은 validation accuracy만 사용한다. Macro F1은 매 epoch 기록하고 각 Child의 최종 선택 head test는 한 번 수행한다. 과거 test-informed 범위 설계의 출처와 미검토 합성 정답에 대한 탐색 한계는 유지한다. 필수 params5개·metrics7개만 tracking하며 warmup/clipping/scheduler/accumulation을 포함한 실제 선택값은 Child configuration.json과 report config 및 Hyperopt document JSON Artifacts에 보관한다. Effective batch는 필수 param에도 실제 accumulation을 반영한다.

## 구현과 검증

`scripts/run_laya_full_experiment.py`의 기존 optional hp.choice 패턴을4개 항목에 확장했다. 모든 값을 float로 변환하던 경로는 NumPy scalar만 native scalar로 변환하도록 바꿔 scheduler 문자열과 accumulation/warmup 정수를 보존한다. 기존 trial_settings의 sampled override 경로가 학습 CLI·MLflow effective batch·JSON 설정에 함께 반영된다. 모델 코드는 변경하지 않는다.

`scripts/prepare_laya_cuda_experiment.py`에서 새 선택 목록의 비어 있음, 정수/실수/문자열 타입·유효 범위·NaN·boolean 오용을 검증한다. 기존 optional 필드가 없는 계획은 같은 검색 공간을 사용한다. 별도 의존성은 추가하지 않았다.

새 통합 테스트는 실제 Hyperopt startup 이후 TPE까지8회를 재생하며 정수/문자열 타입 보존, 학습 override 전달, effective batch와 필수 tracking allowlist, JSON roundtrip 및 test값을 바꿔도 다음 후보가 동일함을 확인한다. 잘못된 신규 선택값의 prepare 거부도 검증한다. 데이터 검증 PASS(130examples/10tools/65isolated scenarios), tracking 통합12개·CUDA 계획6개 PASS. 전체 테스트171개 PASS. 코드 수정 후 graphify update . AST 갱신을 완료했다.

## 재현

```sh
PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
 --plan configs/training/laya-ten-tools-train600-joint20-20261021.json \
 --model-config configs/training/laya-ten-tools-train600-joint20-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-joint20-20261021/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
 --python /home/persona-ai/envs/jery-cuda/bin/python \
 --model-config configs/training/laya-ten-tools-train600-joint20-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-joint20-20261021/tracking \
 --wait-for-run 6261a1bdca194867a835cbfb2216493f --detach
```

Upstream이 FINISHED일 때만 Child를 생성한다. waiting 중에는 새 GPU 학습이 없다. 실패/중단된 upstream이면 오류를 기록하고 자동 학습을 시작하지 않는다. 대기 등록은 실제 학습 시작/완료와 다르다. 기존 state/lock/process를 확인하고 supervisor를 중복 실행하지 않는다.
