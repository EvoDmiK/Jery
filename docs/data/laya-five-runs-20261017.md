# Jery: Laya 5회 반복 학습 — 2026-10-17

새 MLflow Experiment [Jery](https://mlflow.dove-nest.com/#/experiments/47)에서 시드 3407~3411의 독립 학습 5회를 완료했다. 첫 파일럿과 같은 32행 학습·12행 검증, head LR 0.0001, 64스텝(8에폭), microbatch 1 × accumulation 4를 사용했다. 매번 같은 원본 Laya에서 시작하며 이전 실행의 학습 가중치를 이어받지 않았다.

| seed | 학습 시간(초) | train 정답 | validation 정답 | MLflow Run | 모델 버전 |
|---|---:|---:|---:|---|---|
| 3407 | 29.92 | 10/32 | 0/12 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/cff652cafd454a49b46d33fca515b34d) | [v2](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/2) |
| 3408 | 28.85 | 10/32 | 0/12 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/17b351a9aade4bd88d504b1131f425a5) | [v3](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/3) |
| 3409 | 29.63 | 10/32 | 0/12 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/5f6c7ccbe8da4a05915ed9794c1aaa72) | [v4](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/4) |
| 3410 | 34.18 | 10/32 | 0/12 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/82faa8a972ba49a68f069c86bc2541c6) | [v5](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/5) |
| 3411 | 27.74 | 10/32 | 0/12 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/8b058f465df249048b282c5b3c3958e4) | [v6](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/6) |

모든 실행에서 31개 학습 헤드 텐서가 갱신되고 인코더 SHA-256은 유지됐다. 복원한 44행의 선택과 확률이 원본 학습 결과와 같음을 검사했다. 학습 시 MLX 최대 메모리는 약 1.51 GB였다.

각 Run의 Metrics에는 `train/loss`, `train/epoch`, `train/head_learning_rate` 등 64스텝 이력과 `epoch/mean_logged_loss` 8개를 기록했다. 스텝 이력은 각 실행 종료 후 원래 로그에서 동기화하고 API로 재조회했다. 검증은 학습 전후 두 지점만 측정했다. 모든 실행은 FINISHED이며, 학습 이력·head·파생 export·원본 v2 데이터셋·소스·실행 로그의 업로드와 SHA-256 재다운로드 검증을 마쳤다.

기본 가중치와 토크나이저는 이전에 검증해 업로드한 불변 아티팩트 `mlflow-artifacts:/46/cb04fedef6fc41859e37ae8aa6c37cc7/artifacts/model/base`를 공통 참조한다. 각 Run의 `model/dependencies.json`에 base revision·SHA-256과 head SHA-256을 기록했다. MLflow pyfunc가 아닌 네이티브 MLX 형식이다.

첫 기동 1건은 실행기가 가상환경 Python 심볼릭 링크를 해석해 bare Python을 선택하면서 mlx import 전에 실패했다. 가상환경 경로를 보존하도록 수정하고 재시도했다. 해당 FAILED 기록과 원인·로그를 남겼으며, 학습을 정상 완료한 실행은 5건이다. 따라서 Experiment에는 FINISHED 5건과 기동 실패 1건이 있다.

미검토 합성 데이터이며 후보 수 편향과 작은 검증 표본의 한계가 있다. 5회 반복을 실제 사용 정확도의 근거로 삼지 않는다. 공개 prediction-v1 연결, LoRA·보정·실제 MCP 실행은 포함하지 않았다.

데이터 버전 `mcp-routing-v2`, 카탈로그 `hermes-mcp-57e701d87790`, 파생 형식 `decision-export-v1`을 유지했다. 로컬 배치 기록은 `outputs/laya-5runs-20261017-01/batch.json`, 모델·측정 결과는 해당 파일에 기록한 `local_id` 경로에 있다. 실행기는 `scripts/run_laya_repeats.py`이며 기존 `scripts/run_local_training.py`의 시간 제한과 절전 방지를 사용한다.

v1·v2 데이터 검증을 통과했다. 실제 학습 5회와 학습 이력 API read-back, 저장·복원과 모든 업로드 파일의 해시를 확인했다. 작업 브랜치는 dev-codex, HEAD bd6d2ccd0e452deff84181748fc84e56df1ccdec이며 이번 변경은 커밋·푸시하지 않았다.
