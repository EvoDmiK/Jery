# 전체 v2 데이터 Laya 학습 — 2026-10-17

사용자가 승인한 64에폭, 세 가지 사전 지정 학습률을 순차 실행한다. 이 문서는 실행 구성과 재현 방법을 기록한다. 학습 완료와 성능은 아래 상태 파일 및 최종 MLflow 결과로 확인한다.

- Experiment: `Jery` (47)
- [Parent run](https://mlflow.dove-nest.com/#/experiments/47/runs/0b6c6b0cb8324410a5866a7f48f6aa33)
- 원본: `mcp-routing-v2`, 카탈로그 `hermes-mcp-57e701d87790`
- 학습/검증/평가: 원본 668/248/246건, 윈도우 4,172/1,592/1,577개
- 원본 후보 도구를 8개씩 나누며 모든 후보가 정확히 한 번 포함된다. 정답에 따라 후보를 고르거나 사례를 버리지 않는다.
- 각 Child: seed 3407, 64에폭, micro batch 1, 누적 4, 학습률 0.0001 / 0.0003 / 0.001
- 각 Child: 에폭당 optimizer step 1,043회, 총 66,752회
- 모델: 동결한 Laya multilingual encoder와 학습 가능한 결정 head. LoRA 실험이 아니다.

마지막 64에폭 모델로 검증·평가한다. 마지막 검증 정확도가 높은 Child를 선택하고 동률이면 낮은 학습률을 선택한다. 평가 분할로 선택하지 않는다. 평가 분할은 이전 실험에서도 관측되었으므로 회귀 평가이며 새로운 홀드아웃 성능으로 해석하지 않는다. 전체 데이터는 미검토 한국어 합성이며 실제 운영 정확도를 입증하지 않는다.

## 추적 구조

Parent에는 모델, 데이터 버전, 원본 분할 크기만 파라미터로 기록한다. 데이터, 카탈로그, manifest, 감사 결과, 전체 설정과 실행 계획은 JSON artifact다. 각 Child는 실제 시작할 때 생성하고 `mlflow.parentRunId`로 연결한다.

Child 파라미터는 `head_lr`, `epochs`, `seed`, `effective_batch_size` 네 개다. 에폭별 메트릭은 `train_loss`, `val_accuracy`, `val_macro_f1`, `val_coverage`; 마지막 평가 메트릭은 `test_accuracy`, `test_macro_f1`, `test_coverage`다. 나머지 지표, 단계별 손실, 예측, 실행 로그, 환경, 코드 및 복원 증거는 JSON artifact에 보관한다. 가중치는 복원 가능한 원래 safetensors 바이너리로 저장한다. 기존 base artifact를 참조하는 `model/dependencies.json`도 함께 올린다.

다중 윈도우 평가에서 도구 후보를 합치는 과정은 정답을 참조하지 않는다. 불확실한 `delegate`는 내부 평가 결과로만 사용하며 오답 및 미커버 사례에 포함한다. 외부 예측 계약은 변경하지 않는다. 윈도우의 `window_none` 비율이 높으므로 윈도우 정확도를 원본 사례 정확도로 보고하지 않는다.

## 실행과 재시작

```sh
python3 scripts/run_laya_full_experiment.py \
  --python /Volumes/Dove-Nest-SSD/app-data/orca/kimdove/home/workspaces/Jery/main/tmp/venv-decision/bin/python \
  --detach
```

환경은 `requirements-decision.txt` 및 모델 보고서에 기록한다. 위 경로는 실제 MLX 환경의 가상환경 Python이며 symlink를 해제해서 실행하면 안 된다. 루트 `src/`를 사용하고 모델 자산은 `models/laya-multilingual/`에서 읽는다. 실행 코드는 각 Child의 `source.json`으로 보관하므로 Git 커밋 전 변경도 추적할 수 있다.

- 전체 상태: `outputs/laya-full-20261017/tracking/state.json`
- 실행기 로그: `outputs/laya-full-20261017/tracking/supervisor.log`
- Child별 에폭 및 단계 기록: `outputs/model/laya-full-20261017/<local_id>/`
- 모델과 체크포인트: `models/laya-full-20261017/<local_id>/`

실행기는 터미널과 독립된 세션에서 순차 학습을 관리하며 각 Child는 최대 48시간 제한과 caffeinate를 사용한다. 동시에 중복 실행하지 못하도록 파일 잠금을 잡는다. 네트워크 일시 장애 시 로컬 학습을 유지하고 업로드를 재시도한다. 전체 실행기가 종료되었다면 기존 프로세스 종료를 확인한 뒤 같은 명령으로 재시작한다. 완료된 Child는 재학습하지 않고, 중단된 Child는 에폭 체크포인트에서 복구한다. 마지막 완료 에폭 이후의 미완료 단계는 다시 실행할 수 있다.

8 optimizer step과 전체 검증을 실행한 M4 속도 점검으로 Child당 약 12시간, 세 설정 합계 약 36시간을 추정했다. 실제 시간은 데이터 순서와 평가 비용에 따라 달라진다. MLX 캐시는 2 GiB로 제한한다. 최초 점검에서 제한 없는 캐시가 20 GiB 메모리 가드에 도달했던 실패도 Parent artifact로 보존한다.

## 검증

원본 v1/v2 검증, 전체 후보 보존 및 정답 비의존 분할 테스트, 검토 감사, 추적 키 제한 테스트를 실행한다. 모델 자체의 체크포인트·재시작·복원 검증은 [모델 작업 기록](../model/laya-full-training-20261017.md)에 기록한다. 독립 감사 요약은 [JSON](../review/full_audit_summary.json)에 있다.

작업 브랜치는 `dev-codex`, 시작 HEAD는 `bd6d2ccd0e452deff84181748fc84e56df1ccdec`다. 이번 학습 구성은 작업 트리 변경이며 이 요청에서 커밋·푸시하지 않았다. 원본 데이터·가중치·대용량 결과는 Git에 추가하지 않는다.

## 시작 확인

2026-10-17 18:33 KST에 실행기를 시작했다(PID 66936). 첫 Child는 `ea77ce896d6f43a1b163c7ef81da96b1`이며 MLflow에서 `RUNNING`, epochs=64, 학습률 0.0001, Parent 연결을 확인했다. 실행 로그에서 원본 668/248/246건과 윈도우 4,172/1,592/1,577개 로드를 확인했다. 나머지 두 학습률은 대기 중이다. 이 시점에는 본학습 완료 성능이 없다.

검증 결과: 데이터 22개, 검토 37개, 모델 27개, 평가 3개, 통합 15개로 총 104개 테스트 통과. v1 130건·v2 1,162건 데이터 검증과 `git diff --check` 통과, graphify 코드 그래프 갱신 완료.
