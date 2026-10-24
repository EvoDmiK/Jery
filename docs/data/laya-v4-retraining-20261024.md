# v4 데이터 재학습 — 2026-10-24

## 실행 상태

2026-10-24 21:21 KST, 회사 RTX 5070 Ti에서 첫 fold의 실제 optimizer step 진행을 확인했다. 전체 5회는 백그라운드에서 순차 실행한다. 완료 성능은 아직 없다.

[MLflow 재학습 Parent](https://mlflow.dove-nest.com/#/experiments/47/runs/79480c4a3be84189906f4bc2924757fb) → [고정 설정](https://mlflow.dove-nest.com/#/experiments/47/runs/9fe16d70fb8c4a49b56125565912d615) → [첫 fold](https://mlflow.dove-nest.com/#/experiments/47/runs/ebb69afb5e2f4d6199c6f7f7cc685af7).

학습 코드 커밋은 `6e009e6` (`dev-codex`, push 완료). Parent의 Dataset 입력에는 기존에 저장한 `hermes-ten-tools-v4/combined/train`을 연결했다. 각 fold의 실제 입력 JSON과 SHA-256은 Parent의 `data/fold-N/`에 있다.

## 실행 구성

개선 데이터와 도구 설명을 함께 적용한 `hermes-ten-tools-v4` 800개를 사용한다. 기존 검증 결과로 정한 최고 설정을 고정하고, CUDA 서버에서 5개 fold를 순차 학습한다. 모든 fold는 동일한 원본 Laya 가중치에서 시작한다.

| 항목 | 설정 |
|---|---|
| 학습 / 검증 | fold당 640 / 160개, 도구별 64 / 16개 |
| 분할 | scenario 그룹을 보존하는 층화 5-fold, seed 3407 |
| LR / 배치 | 0.00028884521694941566 / 4 |
| Epoch / 조기 종료 | 최대 32, 최소 8, patience 6, min_delta 0.004 |
| Scheduler / Warmup | cosine / 0 |
| Weight decay / Label smoothing | 0 / 0.1 |
| Gradient clipping | 2.0 |
| 카탈로그 | hermes-ten-tools-0b320e20bf3f |
| MLflow 구조 | 재학습 Parent → 고정 설정 → fold 1~5 |

추가 대조 문장 쌍은 서로 다른 정답을 가져도 같은 scenario로 묶는다. 혼합 라벨 그룹은 정답 도구별 개수 조합으로 층화한다. 각 조합의 독립 그룹이 fold 수보다 적으면 분할을 거부한다. 기존 단일 라벨 데이터의 분할 알고리즘은 유지한다. 새 의존성은 없다.

## 평가 해석

800개 모두 한 번씩 검증에 포함된다. 기존 validation 20개는 CV에 사용하지 않고 test 20개는 모델 선택에서 제외한다. 데이터와 설명을 동시에 바꾼 이번 실행으로 두 변경의 개별 효과를 분리할 수는 없다. 이전 600개 CV와 검증 모집단도 달라 점수 차이를 직접적인 개선 효과로 단정하지 않는다. 데이터는 합성이며 사람 검토 대기 상태다.

핵심 파라미터 10개와 에폭별 학습·검증 메트릭만 추적한다. 상세 설정·데이터·분할 해시·보고서는 JSON Artifact로, 모델 체크포인트는 기존 학습 파이프라인으로 저장한다.

## 재현

```sh
python3 scripts/build_laya_stratified_folds.py --source outputs/laya-ten-tools-v4-20261024/combined-controls/augmented --output outputs/laya-ten-tools-v4-stratified5-20261024
/home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_cv_experiment.py --plan configs/training/laya-ten-tools-v4-best-cv5-20261024.json --model-config configs/training/laya-ten-tools-v4-best-cv5-20261024-model.json --state-dir outputs/laya-ten-tools-v4-best-cv5-20261024/tracking --python /home/persona-ai/envs/jery-cuda/bin/python --detach
```

분할 출력이 이미 있으면 덮어쓰지 않는다. 실행 상태는 `outputs/laya-ten-tools-v4-best-cv5-20261024/tracking/state.json`, 로그는 같은 폴더의 `supervisor.log`에 기록한다.

검증: 기존 데이터 130개 통과, 데이터 테스트 38개 통과. CV·계획 테스트는 로컬 16개 통과·Hyperopt 의존 테스트 2개 건너뜀. CUDA 환경에서는 18개 모두 통과했다. 과거 train600의 fold 배정이 그대로 재현되는 것도 확인했다. 로컬과 CUDA 서버의 새 folds.json SHA-256은 `61028a5ef74219c82e1f035a93fe777629e4e76d76ec80c80aa3924f64c135ae`로 동일하다.
