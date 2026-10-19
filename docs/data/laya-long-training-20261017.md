# Jery: Laya 장기 학습률 비교 — 2026-10-17

같은 미검토 합성 train 32행·validation 12행에서 시드 3407을 고정하고, 기존 64스텝(8에폭)을 512스텝(64에폭)으로 늘렸다. head LR 0.0001 / 0.0003 / 0.001 세 설정을 사전에 고정해 비교했다. 매 실행은 동일한 원본 Laya에서 시작했다. 학습 본체를 고정하고 head·scorer·type_emb만 갱신했다. microbatch 1, accumulation 4, warmup 4, linear scheduler, checkpointing 등 나머지 설정은 유지했다.

| head LR | train 정답 | validation 정답 | validation 도구 선택 | validation soft CE | 학습 시간(초) | MLflow Run | 모델 버전 |
|---|---:|---:|---:|---:|---:|---|---|
| 0.0001 | 32/32 | 0/12 | 0/4 | 7.7964 | 228.32 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/8547967dab2b4210aa506b6daee2903d) | v7 |
| 0.0003 | 32/32 | 1/12 | 0/4 | 21.8880 | 239.51 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/f012deac84d04530b15d640b85f17903) | v8 |
| 0.001 | 31/32 | 0/12 | 0/4 | 16.1973 | 236.71 | [실행](https://mlflow.dove-nest.com/#/experiments/47/runs/414415d137dc457f803131e42102b02c) | v9 |

표의 validation soft CE는 기존 평가 함수와 같이 확률을 최소 1e-12로 제한한 값이다. 같은 작은 검증셋을 반복 관찰했으므로 한두 건의 정답 증가로 설정을 우월하다고 선택하지 않는다.

각 실행의 `train/loss`, `train/epoch`, `train/head_learning_rate`에는 512스텝, `epoch/mean_logged_loss`에는 64개 에폭 평균 로그 loss를 저장하고 API로 재조회했다. `live/` 키는 실행 중 run.log를 5초 간격으로 읽은 약 8스텝 간격의 진행 지표다. `live/loss_4dp`는 소수 넷째 자리로 반올림된 로그 값이며 종료 후의 정밀 `train/loss`와 구분한다. 검증은 실제 측정한 학습 전·후 두 지점만 기록했다.

학습 헤드, 데이터셋, 소스, 설정·실행 로그·예측 결과를 각 Run에 업로드하고 파일별 재다운로드 SHA-256을 확인했다. 31개 학습 텐서가 전부 바뀌었고 인코더 해시는 유지됐다. 저장·복원한 44행의 선택과 확률이 정확히 같았다. 기본 가중치·토크나이저는 이전에 검증한 불변 아티팩트를 공유하며 각 `model/dependencies.json`에 URI·revision·SHA-256을 기록했다.

학습 정답 상승과 검증 정답·loss는 구분한다. train 32/32만으로 일반화 개선을 주장하지 않는다. 같은 validation 12행은 이전 실험에서도 이미 관찰했다. 이 비교는 합성 데이터의 학습 경로와 튜닝 탐색이며 실제 사용 정확도·독립 테스트 성능·보정 품질을 입증하지 않는다. 후보 수 편향과 부족한 비선택 시나리오는 여전히 남는다. 학습 시간을 늘려도 데이터 부족과 표현 문제를 해결했다고 볼 수 없다. 공개 prediction-v1 연결과 실제 MCP 호출은 포함하지 않았다.

데이터 버전 mcp-routing-v2, 카탈로그 hermes-mcp-57e701d87790, export decision-export-v1을 유지했다. 사전 계획은 `configs/training/laya-long-20261017.json`, 배치·파일 해시 기록은 `outputs/laya-long-20261017-01/batch.json`이다. 실행기 `scripts/run_laya_repeats.py`는 기존 로컬 실행기의 절전 방지와 20분 시간 제한을 사용한다.

v1 130행·v2 1,162행 데이터 검증, 로컬 실행기 테스트 4개, CLI 확인 및 git diff --check를 통과했다. 작업은 dev-codex HEAD bd6d2ccd0e452deff84181748fc84e56df1ccdec의 미커밋 변경이며 푸시하지 않았다.
