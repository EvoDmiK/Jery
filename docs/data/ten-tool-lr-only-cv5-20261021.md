# 첫 LR 단독 5-fold 실험 · 중단 기록

**상태: 사용자 요청으로 중단.** 이 문서는 이전 실험의 기록이다. 이후 설정을 변경해 시작한 실험은 [현재 LR 탐색 문서](lr-only-reference-lr1e3-20261021.md)를 따른다.

## 실행 결과

| 항목 | 결과 |
|---|---|
| 계획 | LR 20개 조건 × 5-fold = 100회 학습 |
| 완료 | 2개 조건의 10개 fold + 세 번째 조건의 3개 fold |
| 중단 지점 | 세 번째 조건의 네 번째 fold, 24에폭 |
| 로컬 상태 | `interrupted` · 결과와 체크포인트 보존 |
| MLflow | 기존 항목이 삭제된 상태여서 상태 갱신 불가 · 복원하지 않음 |

중단 당시 활성 학습 프로세스가 없고 추가 fold가 시작되지 않았음을 확인했다. 아래 명령은 재현 참고용이며 자동 재개 지시가 아니다.

## 당시 설정

직전 공동 탐색에서 validation accuracy가 가장 높은 Child를 기준으로 LR 외 조건을 고정했다. 이후 사용자가 지정한 `lr1e3` 실험의 Child와는 다른 기준이다.

| 파라미터 | 값 |
|---|---|
| LR | log-uniform `1e-3 ~ 5e-3` · Hyperopt TPE 20개 · 초기 탐색 6개 |
| Warmup / clipping | 75 steps / 2.0 |
| Scheduler / weight decay / label smoothing | `cosine` / 0.1 / 0.0 |
| 유효 배치 / seed | 4 / 3407 |
| 조기 종료 | 최대 32에폭 · 최소 8에폭 · patience 6 · min_delta 0.004 |

기준 Child: `c0635c4ce7c44e97aad5e9f695516a0b` · 이전 validation accuracy 0.85. 기준 선택에 test는 사용하지 않았다.

## 검증 방법

- 합성 train 600개만 분할: fold별 학습 480개 / 검증 120개, 도구당 검증 12개.
- 같은 `scenario_id`를 분할 간에 섞지 않고, 모든 LR 조건에 동일한 5-fold 적용.
- 기존 validation 20개는 제외. Test 20개는 최종 평가에만 사용.
- 탐색 기준은 fold별 best-trained validation accuracy의 평균인 `cv_search_accuracy`.
- Test 점수는 fold 모델 5개의 평균이며 전체 600개로 학습한 단일 모델의 점수가 아님.

층화와 그룹 보존은 표준 라이브러리로 구현했다. 상세 지표 정의와 데이터 분할 규칙은 [CV 계약](../contracts/laya-cross-validation-v1.md)을 따른다.

## 구현·검증 근거

[Fold builder](../../scripts/build_laya_stratified_folds.py)와 [CV 실행기](../../scripts/run_laya_cv_experiment.py)를 추가하고 기존 학습·업로드·복구 흐름을 재사용했다. 구현 당시 전체 테스트 181개, 실제 Hyperopt 후보 8개의 비-LR 불변식 검사, 기존 데이터 검증을 통과했다.

Fold 배정 SHA: `5e823b9408bf8c410005042315c362b8291aed9efd3e461cec7cc79fbac33e9b`
이전 Parent: `790b62ac777c43098b47cadd134d3a4e`

## 당시 실행 명령

```sh
python3 scripts/build_laya_stratified_folds.py \
 --source outputs/laya-ten-tools-training-v3-final-20261020/augmented \
 --output outputs/laya-ten-tools-stratified5-20261021 --n-splits 5 --seed 3407
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_cv_experiment.py \
 --plan configs/training/laya-ten-tools-train600-lr-only-cv5-20261021.json \
 --model-config configs/training/laya-ten-tools-train600-lr-only-cv5-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr-only-cv5-20261021/tracking \
 --python /home/persona-ai/envs/jery-cuda/bin/python --detach
```
