# 10개 도구 단일 선택 학습 진단

사용자가 한국어 의도 분류는 별도 프로젝트에서 진행하고, Jery는 도구를 10개로 줄여 학습하도록 요청했다. 원본 데이터와 96개 도구 실험을 보존하고 새 `hermes-ten-tools-v1` / `decision-export-v2` 과제로 분리한다. 기존 96개 도구 공동 탐색은 5회 모두 완료됐으며 최고 trained validation 16.94%였다. 이 수치는 새 10개 고정 후보 과제와 직접 비교하지 않는다.

## 도구와 표현

Obsidian의 vault·edit·graph·dataview·bases와 Swallow의 get_stock_price·get_price_history·get_portfolio·order·backtest를 사용한다. 각 요청에 정확히 10개 후보를 한 번에 제시한다. state는 요청·호출 전 context, choice criteria는 별칭과 실제 기능을 설명하는 한국어 문장이다. 별도 후보 카드와 창 분할·재대결은 없다. `no_tool`, `clarify`, `unsupported` 판단과 action 예측은 이번 과제에 포함하지 않는다. 실제 도구를 실행하지 않는다.

설명은 등록 카탈로그와 대조했다. 예를 들어 get_stock_price에는 현재가뿐 아니라 개별 종목 보유 현황이 포함되고, get_portfolio는 계좌 전체 자산·보유목록 조회다. Dataview 문법 검사와 Bases YAML 관리 기능도 선택지에 반영했다.

## 데이터와 단계

고정 후보로 바꾸면서 중복이 된 request+context를 먼저 제거했다. 원본 시나리오 분할 경계를 유지하고 도구별 수를 동일하게 맞췄다. metadata에 원 후보·허용 정답·출처를 보존하며 입력에 넣지 않는다.

| 단계 | train | validation | test | 목적 |
|---|---:|---:|---:|---|
| memorization | 20 (도구별 2) | 20 | 20 | 작은 학습 집합을 실제로 학습하는지 |
| full | 60 (도구별 6) | 20 | 20 | 같은 10개 도구에서 학습·평가 |

두 profile의 validation/test는 선택지 순서까지 동일하다. 첫 단계는 LR 2e-5·warmup 0·weight decay 0으로 최대 64에폭을 관찰하며 매 epoch train 정확도를 JSON에 기록한다. 마지막 학습 head의 train 정확도 95% 이상을 확인한 뒤에만 full 단계의 5회 Hyperopt LR 탐색을 시작한다. train 성능은 학습 경로 진단용이며 head 선택·검색 점수로 쓰지 않는다. validation으로 최고 trained head를 정하고 serving baseline은 별도로 보호한다. test는 각 Child 최종 serving head에 한 번 평가한다.

full 단계의 최종 LR 범위는 `1e-4~1e-3`, 최대 64에폭·min_epochs 32·patience 5다. 낮은 LR의 20개 train fit 실패와 높은 LR의 학습 성공에 따라, 기존 초안 `1e-6~3e-5`·min_epochs 10을 실행 전에 조정했다. window_none 사례가 없어 관련 가중치는 비활성 1로 고정한다. 다른 설정과 분할은 동일하다. 필수 Child params 5개·metrics 7개를 유지하며 나머지는 JSON artifact다. 이 과제의 `val_macro_f1`/`test_macro_f1`은 명시적으로 10개 도구의 macro F1이며 기존 decision F1과 구분한다.

## 검증과 재현

원본 v1/v2 검증과 단일 선택 계약 검사를 수행한다. native builder·tokenizer로 skip, target 인덱스/확률, 학습·추론 특징 불일치, 입력 잘림·충돌을 검사한다. 기존 창 한도 8·null 균형·지표 동작은 유지하고 명시적 ten_tool_choice 설정에서만 10개 단일 선택을 사용한다. 의존성은 기존 `requirements-decision-cuda.txt`다.

회사 서버 `/home/persona-ai/git/Jery-cuda`에서 전용 Python `/home/persona-ai/envs/jery-cuda/bin/python`을 사용한다.

```sh
python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-ten-tools-memorization-20261020.json \
  --model-config configs/model/laya-ten-tools-20261020.json \
  --state-dir outputs/laya-ten-tools-20261020/memorization-tracking
TORCHINDUCTOR_COMPILE_THREADS=4 python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/model/laya-ten-tools-20261020.json \
  --state-dir outputs/laya-ten-tools-20261020/memorization-tracking --detach
```

full 단계는 같은 명령의 plan을 `laya-ten-tools-full-20261020.json`, model-config를 `laya-ten-tools-full-20261020.json`, state-dir을 `outputs/laya-ten-tools-20261020/full-tracking`으로 바꾸고, 먼저 진단 통과 증거를 확인한다. 원본·파생 데이터와 가중치를 덮어쓰지 않는다.

합성 정답은 미검토 상태이며 작은 검증·평가 집합의 점수는 탐색 결과다. 시나리오가 분리돼도 기존에 관찰된 test이므로 독립적인 실제 사용 성능 근거가 아니다. 저장소의 dataset README와 [v2 계약](../contracts/decision-export-v2.md)에 한계를 기록한다.

## 2026-10-20 실행 증거

회사 RTX 5070 Ti에서 첫 진단 Parent `17ebcc82ff7d43fc9d4d7d007cc4d29c` / Child `659bae0e7bbb480a9c0e72c431475cb7`가 64에폭·320 optimizer step을 완료했다. 학습 호출은 59.31초, CLI는 76.63초였으며 head 31개 텐서가 갱신되고 encoder는 유지됐다. LR 2e-5의 마지막 train 정확도는 70%로 95% gate를 통과하지 못했다. Validation은 초기·학습 head 모두 50%로, serving은 초기 head를 유지했다. 따라서 60개 full Hyperopt는 시작하지 않았다.

학습 집합 fit만을 근거로 LR 1e-3의 별도 64에폭 진단을 추가했다. 재현 시 위 plan을 `configs/training/laya-ten-tools-memorization-lr1e-3-20261020.json`, state-dir을 `outputs/laya-ten-tools-20261020/memorization-lr1e-3-tracking`으로 바꾼다. Parent는 `37611c82fb924ba6bccb5d4410db9226`이며 기존 결과를 덮어쓰지 않는다. Gate 결과와 재시도 근거는 두 Parent의 `diagnostic/initial-train-fit-gate.json`에 업로드하고 SHA-256 read-back을 확인했다. Test는 gate나 LR 조정 근거로 사용하지 않았다.

Claude가 native CUDA 단일 10개 선택과 도구별 F1·train fit 보고를 구현했고 AGY가 원본 무결성·재현성·분할·중복·입력 누수를 독립 감사했다. 통합 후 학습용 Python에서 전체 145개 테스트가 통과했다. 두 profile의 native builder/tokenizer readiness는 skip·target 불일치·학습/추론 특징 불일치·잘림 모두 0건이며 최대 229 token이다.

재진단 Parent `37611c82fb924ba6bccb5d4410db9226` / Child `c997b44b16b04a9c9f43c02725ef616e`가 64에폭을 완료했다. 마지막 train 정확도는 100%로 gate를 통과했고, 최고 trained validation은 epoch 18의 60%(초기 50%)였다. 마지막 train fit과 validation 선택 head는 서로 다른 평가 대상이다. 학습 head 재로딩 후 validation 예측이 동일하며 encoder는 유지됐다. 학습 호출 59.91초, CLI 77초 안팎이다. Gate JSON은 진단 Parent에 업로드·해시 검증했고, 본 실험은 별도 Parent `c328674ecd9c4eabb1bf3d4dbc6c9dde`로 준비했다.
