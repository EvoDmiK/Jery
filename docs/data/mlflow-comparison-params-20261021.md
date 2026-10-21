# MLflow 파라미터 비교 설정

Warmup, clipping, scheduler, weight decay, label smoothing을 MLflow 표에서 직접 비교할 수 있도록 추가했다. 전체 설정과 에폭 이력은 JSON Artifact로 보관한다.

## 표시 항목

기존 핵심 파라미터 5개에 아래 5개를 더해 **총 10개**를 기록한다. 기존 지표 7개는 유지한다.

| MLflow key | 의미 |
|---|---|
| `warmup_steps` | Warmup optimizer step 수 |
| `max_grad_norm` | Gradient clipping norm |
| `lr_scheduler` | 학습률 scheduler |
| `weight_decay` | AdamW weight decay |
| `label_smoothing` | Label smoothing 계수 |

**Runs → Columns → Parameters**에서 위 항목을 선택하면 LR·배치와 함께 비교할 수 있다.

## 기존 기록 복원 결과

| 항목 | 결과 |
|---|---:|
| 대상 Child | 105개 |
| 추가한 파라미터 종류 | 5개 |
| 추가한 값 | 525건 = 105 × 5 |
| 오류 / 확인하지 못한 필드 | 0 / 0 |

525개 종류를 추가한 것이 아니라, 105개 실행에 같은 5개 항목을 채웠다. 실제 MLflow 값을 다시 읽어 검증했다.

## 복원 규칙

[동기화 스크립트](../../scripts/sync_laya_comparison_params.py)는 각 실행의 `configuration.json`, 없으면 `details/report.json`에서 실제 사용값을 읽는다.

- 값이 없으면 기본값으로 추정하지 않는다.
- 기존 값과 충돌하면 덮어쓰지 않고 오류로 남긴다.
- 같은 값으로 재실행하면 중복 기록하지 않는다.
- 학습 상태·지표·모델은 변경하지 않는다.

공동 탐색 중에는 별도 observer가 새 Child를 확인해 기록을 보강했다. 학습은 observer와 독립적으로 실행됐다.

## 재현과 검증

```sh
PYTHONPATH=scripts python3 scripts/sync_laya_comparison_params.py \
  --output outputs/laya-comparison-params-20261021/history-sync.json
```

상세 결과는 위 JSON에 실행별 출처·기대값·추가값으로 저장했다. 변경 당시 전체 테스트 174개와 데이터 검증을 통과했고, 충돌 거부·누락값 미추정·멱등성을 확인했다. 추가 의존성은 없다.

이후 완료 검증에서 발생한 충돌과 해결은 [복구 기록](mlflow-param-backfill-recovery-20261021.md)에 정리했다.
