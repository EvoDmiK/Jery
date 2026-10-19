# Laya 첫 로컬 학습 파일럿 — 2026-10-17

## MLflow 학습 이력 보완

최초 업로드는 단계별 데이터를 `evaluation/steps.json` 아티팩트로만 보관하고 Metrics에는 최종 요약값만 기록했다. 원래 로그에서 각 실행의 64스텝 loss·epoch·grad norm·메모리·학습률과 8에폭의 평균 로그 loss를 Metrics로 보완했다. 첫 실행은 `train/loss`, `train/epoch`, `train/head_learning_rate`, `epoch/mean_logged_loss`에서 볼 수 있다. 두 번째 실행은 같은 키 앞에 `diagnostic_lr1e3/`가 붙는다. 각 실행 400개 기록을 API로 다시 읽어 검증했으며 재실행 시 중복 추가하지 않는다.

에폭 loss는 그 에폭의 optimizer-step 로그 loss 평균이다. 검증값은 실제 측정한 학습 전·후 두 지점만 `eval/validation/`에 기록했다. head 학습률은 로그의 고정 encoder 그룹 학습률에 공통 scheduler 비율을 적용해 복원했다. 시각은 초 단위 run.log의 첫 스텝 시각과 원래 경과 시간으로 복원해 약 1초 미만의 기준 시각 오차가 가능하다. 재학습하지 않았다.

```sh
python3 scripts/sync_laya_training_metrics.py \
  --report-dir outputs/model/laya-pilot-20261017/head-only-s3407-try1 \
  --run-id cb04fedef6fc41859e37ae8aa6c37cc7
```

Mac mini M4 32 GiB에서 Laya Multilingual의 헤드 학습, 저장·복원과 추론이 동작했다.
검증 12건의 내부 선택 정답은 학습 전후 모두 0건이므로 성능 개선은 확인하지 못했다.
이번 결과는 학습 경로의 실행 가능성 확인이며 운영 모델 채택이나 정확도 평가가 아니다.

## 실행 범위와 역할

Codex는 `decision-export-v1` 변환과 공통 계약·통합·실험 기록을 맡았다.
Claude는 `dev-claude`의 모델 코드, 실제 MLX 학습과 저장·복원을 맡았고,
Antigravity는 `dev-agy`에서 원본 대조와 후보 창·분할·출처의 독립 감사를 맡았다.

원본은 `mcp-routing-v2`, 카탈로그는 `hermes-mcp-57e701d87790`이다.
train 32행(26 시나리오 그룹), validation 12행(10 그룹)을 사용했으며 그룹 교집합은 없다.
선택 정답을 내부 창에서 제외한 음성 사례는 각각 4행과 2행이다.
전부 synthetic·needs_human_review·execution_verified=false이며 실제 MCP를 호출하지 않았다.
액션 예측과 공개 prediction-v1 연결은 이번 범위에 포함하지 않았다.
`window_none`은 내부 창의 정답으로, 공개 `unsupported`로 바로 반환할 수 없다.

## 실제 실행과 결과

- 모델: `convaiinnovations/laya-multilingual`, revision `1720e3e3357cfe1e281542e223f8273b0890ca34`.
- 학습 방식: 인코더 고정(float16 텐서 89개와 float32 텐서 45개), head·scorer·type_emb 14,770,945개 파라미터 학습(전체 321,710,593개).
- seed 3407, 시퀀스 1024, micro batch 1, accumulation 4, 64 optimizer steps(8 epochs), head LR 0.0001.
- 환경: Python 3.12.15, Unsloth/Unsloth Zoo 2026.10.6, MLX/MLX Metal 0.32.3, Transformers 5.17.0.
- 학습 33.03초, 전체 측정 68.53초. 학습 중 MLX peak 1.508 GB, 복원 포함 MLX peak 2.251 GB, 프로세스 최대 RSS 5.512 GB. 각각 다른 측정치이며 장치 전체 메모리 사용량이 아니다.
- 실제 Laya 입력 최대 토큰: train 898, validation 767. 옵션·지침·state 잘림과 옵션 충돌은 44행 모두 0건.
- 64스텝 loss는 모두 유한하고 학습 대상 31개 텐서가 모두 변경됐다. 고정 인코더의 학습 전후 SHA-256은 동일하다.
- train 내부 top-1 any-of 정답 10/32 → 10/32, validation 0/12 → 0/12.
- 동일 행의 추론 soft CE는 train 2.3869 → 2.0323, validation 4.1325 → 3.1222. 손실 감소를 정확도 개선으로 해석하지 않는다.
- 학습 헤드 59,086,741 bytes 저장. 복원한 44행의 선택이 전부 동일하고 확률 최대 절대 차이는 0이다.
- 복원 모델 warm 추론 60회: p50 82.71 ms, p95 106.45 ms. 입력 길이와 처리 범위가 다른 기존 기준 모델과 직접적인 우열 비교는 하지 않는다.

두 번째 진단 실행도 보존했다. 동일 seed·분할·64스텝에서 head LR만 0.001로 바꾼
`head-only-s3407-lr1e-3-try2`는 학습 27.95초, train 내부 정답 10/32 → 16/32,
validation 0/12 → 0/12였다. 저장·복원 확률 차이는 0이었다.
이 실행을 제외해 결과를 좋게 보이게 하거나 검증 성능 개선으로 주장하지 않는다.
추가 학습·튜닝은 진행하지 않았다.

## 산출물과 재현

- [변환 계약](../contracts/decision-export-v1.md), [공동 전략](decision-training-strategy.md)
- [모델 실행·환경·진단 보고서](../model/laya-pilot-20261017.md)
- [독립 감사 보고서](../review/laya-pilot-audit-20261017.md)
- 기본 모델: `models/laya-multilingual/`
- 학습 헤드: `models/laya-pilot-20261017/head-only-s3407-try1/trainable.safetensors`
- 동결 export: `outputs/laya-pilot-20261017/export/`
- 학습 리포트·단계별 loss·입력 예산·전후/복원 예측: `outputs/model/laya-pilot-20261017/head-only-s3407-try1/`

원본과 export의 SHA-256, 실행 명령, 학습 환경·설정은 manifest 및 report.json에 기록했다.
가중치와 대용량 결과는 Git에 넣지 않는다. 완료한 실행의 설정 스냅샷은 보존하고,
새 실험은 새 출력 디렉터리로 실행한다. 데이터 생성과 학습 재현 명령은 계약 및 모델 보고서를 따른다.

## 남은 한계와 다음 실험 조건

32행으로는 한국어 도구 분류의 일반화나 확률 보정을 판단할 수 없다.
현재 합성 후보 수 편향, 부족한 비선택 시나리오, 압축 카드의 의미 손실 가능성이 남는다.
이번 입력 창은 원본 후보의 부분집합이며 OOF 검색기의 결과가 아니다.
검증 12행을 반복 튜닝하는 대신 합의된 별도 검토 데이터와 분할을 준비해야 한다.

다음에는 후보 수를 맞춘 데이터와 명세 표현을 검토하고, 동일 학습 데이터로 다시 학습한
선형 기준 모델과 Laya를 비교한다. 헤드만으로 부족할 때 인코더 LoRA를 별도 실험한다.
독립 calibration/fresh test 없이 운영 정확도나 자동 승인 기준을 정하지 않는다.

## 실험 추적

[MLflow 실행](https://mlflow.dove-nest.com/#/experiments/46/runs/cb04fedef6fc41859e37ae8aa6c37cc7)에 기본 모델·토크나이저·학습 헤드·원본 및 파생 데이터셋·측정 결과를 저장했다. [Jery-LayaHead v1](https://mlflow.dove-nest.com/#/models/Jery-LayaHead/versions/1)은 네이티브 MLX 형식이며 운영 배포하지 않았다. 업로드·재다운로드 SHA-256 검증 기록은 `outputs/laya-pilot-20261017/mlflow-upload/receipt.json`에 있다. 두 번째 진단 실행은 별도 additional_attempts 경로에 보존한다.

## 통합 검증 및 Git 상태

`python3 scripts/validate_dataset.py`(130행) 및 `python3 scripts/validate_mcp_dataset.py`(1,162행)가 통과했다. `python3 -m unittest discover -s tests/<분야> -v` 기준 data 20개, integration 13개, model 20개, evaluation 3개, review 22개로 관련 테스트 총 78개가 통과했다. 후보 밖 선택 거부와 정답 메타데이터 입력 배제 검증도 포함한다. CLI `--help`, `git diff --check`, 새 파일 공백 검사 및 `graphify update .`를 확인했다.

통합 브랜치는 `dev-codex`, 현재 HEAD는 `bd6d2ccd0e452deff84181748fc84e56df1ccdec`이며 이번 변경은 커밋·푸시하지 않았다. 실행 당시 워커의 HEAD와 dirty 상태는 원래 report.json에 별도로 보존한다. 통합 코드와 검증 기록은 `integration-verification.json`에 저장한다.
