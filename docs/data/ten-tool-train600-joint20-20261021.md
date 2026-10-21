# Train 600 · 다중 파라미터 탐색 20회

**20회 모두 완료.** LR, 정규화, warmup, 배치, clipping, scheduler를 함께 탐색한 실험이다.

[MLflow 실험 보기](https://mlflow.dove-nest.com/#/experiments/47/runs/3b7fd13fe2f74146a520fa64d0c2305d) · [현재 LR 단독 실험](lr-only-reference-lr1e3-20261021.md)

## 탐색 범위

| 항목 | 후보 |
|---|---|
| Head LR | log-uniform `1e-3 ~ 5e-3` |
| Weight decay | 0, 0.05, 0.1 |
| Label smoothing | 0, 0.1 |
| Warmup steps | 10, 75, 150 |
| Gradient accumulation | 1, 2, 4 |
| 유효 배치 | micro batch 4 × accumulation → 4, 8, 16 |
| Gradient clipping | 0.5, 1.0, 2.0 |
| Scheduler | `linear`, `cosine` |

Hyperopt TPE로 20개 조건을 비교했다. 초기 탐색은 6개다. 여러 변수를 함께 바꾼 거친 탐색이므로 개별 파라미터의 효과를 분리해 판단할 수는 없다. Warmup은 에폭 비율이 아닌 optimizer step 수이며, 배치가 달라지면 전체 step 수도 달라진다.

## 고정 조건과 평가

| 항목 | 설정 |
|---|---|
| 데이터 | 합성 `hermes-ten-tools-v3` · train 600 / validation 20 / test 20 |
| 학습 | Laya 인코더 고정 · head/scorer/type embedding 학습 |
| Seed | 3407 |
| 종료 정책 | 최대 32에폭 · 최소 8에폭 · patience 6 · min_delta 0.004 |
| 환경 | Linux · RTX 5070 Ti · 매 조건 동일한 base에서 시작 |

TPE와 체크포인트 선택에는 validation accuracy를 사용했다. Test는 선택한 head의 최종 평가에 사용했다. 다만 과거 test 결과를 참고해 탐색 범위를 정한 이력이 있어 독립적인 최종 성능 검증으로 해석하지 않는다.

실제 선택값과 전체 설정은 Child의 `configuration.json`, `details/report.json`, Hyperopt JSON Artifact에 남겼다. MLflow 표에 표시하는 항목은 [파라미터 비교 설정](mlflow-comparison-params-20261021.md)을 따른다.

## 결과와 한계

검증 정확도가 가장 높은 best-trained Child는 `c0635c4ce7c44e97aad5e9f695516a0b`였으며 **0.85**를 기록했다. Warmup 75, clipping 2.0, cosine, weight decay 0.1, label smoothing 0을 사용했다.

검증 사례가 20개인 합성 데이터 반복 실험이다. 실제 사용자 요청에 대한 성능 개선으로 단정하지 않는다. 후속 5-fold 실험은 별도 문서에서 관리한다.

## 구현과 검증

기존 실행기에 선택형 탐색 항목을 추가하고, 문자열·정수 타입을 학습 설정과 Artifact까지 보존했다. 새 선택값의 범위·타입 검사와 TPE 이후 후보 검증도 추가했다. 모델 코드와 의존성은 변경하지 않았다.

구현 당시 전체 테스트 171개와 데이터 검증을 통과했다. 실제 Hyperopt 후보 8개로 타입 보존·설정 전달·JSON 왕복과 test 값의 탐색 배제를 확인했다.

실행 중 발생한 완료 검증 오류는 [같은 Parent에서 복구](mlflow-param-backfill-recovery-20261021.md)했다.

## 당시 실행 명령

아래는 최초 등록 명령이다. 이미 완료된 실험을 다시 실행하기 전에 기존 state와 lock을 확인해야 한다.

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
