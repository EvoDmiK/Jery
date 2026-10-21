# MLflow 비교용 optimizer 파라미터 표시 — 2026-10-21

사용자가 Warmup·gradient clipping·scheduler·weight decay·label smoothing을 MLflow 표에서 비교하기 어렵다고 지적했다. 기존 JSON-only 정책에서 아래5개를 비교 필수 params로 승격한다. 기존 필수5개는 유지하고 metrics7개는 바꾸지 않는다.

| 표의 param key | 실제 설정 |
|---|---|
| warmup_steps | optimizer warmup step 수 |
| max_grad_norm | gradient clipping norm |
| lr_scheduler | linear/cosine 등 실제 scheduler |
| weight_decay | AdamW weight decay |
| label_smoothing | 사용자가 weight smoothing으로 지칭한 label smoothing |

MLflow Runs 표의 Columns/Parameters에서 위 key를 선택하면 LR·effective_batch_size와 함께 비교할 수 있다. 전체 설정·epoch 이력 JSON Artifact도 보존한다.

## 코드와 실행 보존

`scripts/run_laya_full_experiment.py`는 기존 핵심 tracking 계약에 허용된 비교 param subset을 추가로 받을 수 있다. `trial_settings`가 sampled override를 우선하고 나머지는 실제 model config에서 읽는다. `configs/training/laya-ten-tools-train600-joint20-20261021.json`에5개 key를 추가해 후속 실행에 직접 기록한다. 이전 계획의 핵심 key 집합과 unknown key 거부는 유지한다.

이미 실행 중인 supervisor와 저장된 plan/config는 바꾸거나 재시작하지 않는다. `scripts/sync_laya_comparison_params.py`가 기존 Child의 immutable configuration.json(없으면 details/report.json)에서 실제 사용값만 읽어 MLflow params를 추가한다. 기존값과 다르면 덮어쓰지 않고 오류를 기록한다. 누락된 설정은 기본값으로 추정하지 않는다. 재실행 시 같은 숫자의 문자열 표현 차이는 허용하고 신규 write가 발생하지 않는다. 추가 후 GET read-back을 검증하고 실행 상태/metrics/model은 바꾸지 않는다.

Jery(47)에서 jery.role=child인 실행을 pagination하여 보강한다. 새 Child를 생성하는 현재 공동 탐색 Parent3b7fd13fe2f74146a520fa64d0c2305d에는 완료까지15초 간격의 별도 observer를 연결한다. 네트워크 실패는 observer가 재시도하고 학습은 독립적으로 계속된다. Backfill은 학습·평가를 다시 수행하지 않는다.

```sh
PYTHONPATH=scripts python3 scripts/sync_laya_comparison_params.py \
 --output outputs/laya-comparison-params-20261021/history-sync.json
PYTHONPATH=scripts /home/persona-ai/envs/jery-cuda/bin/python scripts/sync_laya_comparison_params.py \
 --watch-parent 3b7fd13fe2f74146a520fa64d0c2305d \
 --output outputs/laya-comparison-params-20261021/joint-watch.json
```

## 검증

데이터 검증 PASS(130examples/10tools/65isolated scenarios). Tracking 통합13개·동기화2개·CUDA 계획6개 PASS. 실제 sampled optimizer 값이 params와 training override 양쪽에 같은 값으로 전달되는지, config fallback, 없는 값 미추정, 충돌 거부와 멱등성을 검증했다. Graphify AST update 완료. 새 의존성은 없다. 실행 중 Child의 params5개가 실제 MLflow GET에 나타나고 기존 FINISHED/RUNNING 상태가 유지됨을 확인했다.

기존 Child105개에서 실제5개 값 모두 복원하고 params525개를 추가했다. 오류0·미확인 field0이며 GET read-back으로 검증했다. 전체 테스트174개 PASS. 상세 결과는 outputs/laya-comparison-params-20261021/history-sync.json에 실행별 출처·기대값·추가값으로 저장한다.
