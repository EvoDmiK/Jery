# MLflow 비교 params 추가 후 supervisor 중단 복구 — 2026-10-21

사용자가 공동 탐색 표가 멈춰 있다고 보고했다. 서버에서 Parent3b7fd13fe2f74146a520fa64d0c2305d의 supervisor226349가 종료됐고 state는 supervisor_failed임을 확인했다. Child1~9는 finished, Child10 f599948302284d26ab2a7a66173d2379는 uploading에서 멈췄다.

## 원인과 수정

Child10의 실제 CUDA 학습은17epoch/2550steps, 2026-10-21 01:47:11 UTC에 succeeded로 종료했다. Report·head·원본 config/source가 보존돼 있다. 선택 epoch11의 test accuracy0.65/F1 0.56도 이미 기록했다. 03:24 UTC 확인 시 supervisor는 죽어 있었고 예외는 `Nonessential MLflow keys were logged, or required keys are missing`였다.

비교용5개 optimizer params를 기존 Child에 backfill했지만 기존 supervisor의 완료 검증은 예전 plan의5개 params와 정확히 같은 key 집합만 허용했다. 실제10개를 만나 실패했다. 파라미터 표시 확장 때 post-upload 완료 검증 경로를 갱신하지 못한 코드 결함이다.

`scripts/run_laya_full_experiment.py`의 완료 key 검증을 `validate_logged_keys`로 분리하고 plan의 필수 params는 계속 요구하되 승인된 COMPARISON_PARAMS5개만 추가 허용한다. Unknown params·필수 누락·metrics 추가/누락은 계속 거부한다. 새 plan에서 비교 params를 필수로 선언했으면 누락을 허용하지 않는다. 실제 실패 Child의 remote params/metrics도 새 검증을 통과했다.

회귀 테스트는 legacy5개/보강10개/신규10개를 정상으로, unknown·core 누락·metric 추가/누락·새 plan의 비교 param 누락을 오류로 검증한다. Tracking 통합14개 PASS, 원본 데이터 검증 PASS(130examples/10tools/65isolated scenarios). Graphify AST update를 수행한다.

## 복구 원칙과 재현

기존 supervisor 사망과 모델 status succeeded/report 존재를 확인한 뒤 동일 state와 Parent로 재실행한다. Prepare를 새로 실행하거나 새 Parent/Child를 만들지 않는다. 완료된1~9는 건너뛰고, Child10은 report가 존재하므로 학습 프로세스를 만들지 않고 업로드·완료 검증만 재수행한다. 이후11~20을 이어 학습한다. 다른 GPU 작업은 변경하지 않는다.

복구 전 tracking/recovery-param-keys-before.json에 Child10의 report/steps/metrics/head/config/source6개 SHA·mtime와 runner-1 목록을 저장했다. 복구 후 같은 값을 비교해 재학습하지 않았음을 검증한다. 원래 source/config와 실제 사용한 plan을 덮어쓰지 않는다.

```sh
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
 --python /home/persona-ai/envs/jery-cuda/bin/python \
 --model-config configs/training/laya-ten-tools-train600-joint20-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-joint20-20261021/tracking --detach
```

이번 재실행은 이미 FINISHED인 upstream 대기를 다시 지정할 필요가 없다. 현재 Parent/Child의 실제 복구 결과와 supervisor PID는 아래에 추가 기록한다. 자동 params observer는 학습과 독립적이며 유지한다. 추가 의존성·모델·데이터 변경은 없다.
