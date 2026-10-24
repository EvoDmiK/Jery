# OOF 오류 분석과 seed 재검증

기준·최고 설정의 OOF 예측 600개를 비교했고, 같은 5개 fold에서 3개 seed의 재검증 30회를 모두 완료했다. [완료 결과와 개선 제안](ponytail-validation-and-performance-20261024.md)을 참고한다. 아래 실행 확인은 시작 당시 기록이다. 실제 요청 평가셋은 후보 13개·사람 검토 완료 0개여서 준비 단계다.

## 완료 실험에서 확인한 차이

분석 대상은 [완료된 LR 고정 실험](https://mlflow.dove-nest.com/#/experiments/47/runs/d9636e10f0b6449dbf6975e945e07aad)의 기준 trial1과 최고 trial9다. 각 fold에서 선택된 head의 저장 예측을 원래 validation 정답에 연결했다. 입력·fold 해시, ID별 1회 포함, 후보 안의 선택, report 정확도 일치를 검사했다.

| 지표 | 기준 trial1 | 최고 trial9 |
|---|---:|---:|
| OOF 정답 수 / 600 | 449 | 461 |
| OOF 정확도 | 74.83% | 76.83% |
| OOF 전체 Macro F1 | 0.7463 | 0.7674 |
| fold별 Macro F1 평균 | 0.7449 | 0.7643 |

OOF 전체 F1은 600개의 혼동행렬로 계산한다. MLflow의 조건별 F1은 5개 fold F1의 평균이므로 두 숫자가 조금 다르다.

- 기준 오답 → 최고 정답: **47개**.
- 기준 정답 → 최고 오답: **35개**.
- 두 설정 모두 오답: **104개**.
- 두 설정 모두 정답: **414개**.

정답은 순증 12개, 정확도는 2.0%p 개선됐다. fold별 정확도 변화는 +3.33, +5.00, −1.67, +4.17, −0.83%p다. 5개 중 3개에서 개선했으며 seed가 바뀌어도 개선되는지는 새 실험으로 확인한다. 이 데이터는 설정·epoch 선택에 사용한 합성 검증 자료이므로 독립 일반화 성능이 아니다.

## 먼저 검토할 오류

| 정답 → 최고 설정의 오답 | 기준 건수 | 최고 건수 | 검토할 경계 |
|---|---:|---:|---|
| vault → edit | 21 | 16 | 읽기·검색과 내용 수정 |
| get_portfolio → order | 15 | 10 | 잔고·보유내역 조회와 주문 |
| get_price_history → get_stock_price | — | 8 | 과거 기간 이력과 현재가 |

vault 정답은 60개 중 26→30개, edit는 38→44개로 개선했지만 여전히 취약하다. bases·backtest·price_history·order는 정답 수가 소폭 줄었다. 파라미터 하나로 모든 도구가 함께 좋아진 결과는 아니다.

`cases.jsonl`에 요청·이전 맥락·정답·두 예측·fold·scenario ID·개선/퇴보 구분을 남겼다. 공통 오답 104개를 먼저 검토하고, 요청과 맥락이 충분한지, 라벨이 명세와 맞는지 확인하는 것이 다음 순서다. 기준·최고 조건의 10개 fold 보고서에서 입력 잘림은 모두 0건이었다. 따라서 현재 오류를 입력 잘림으로 설명할 근거는 없다. 모델이 틀렸다는 이유로 라벨을 자동 수정하지 않았다.

## 재검증 설정 · 30회 완료

[MLflow Parent 열기](https://mlflow.dove-nest.com/#/experiments/47/runs/af613b715d804393b17872be8d5ad671)

실험 이름은 `laya-ten-tools-seed-confirmation-cv5-20261024`다. `Parent → 설정·seed 6개 → fold 5개`로 기록하며, 총 **30회 학습**을 순차 실행한다. seed3407도 새로 실행해 이전 결과의 재현성을 확인한다.

| 항목 | 기준 설정 | 최고 설정 |
|---|---|---|
| LR | 0.00028884521694941566 | 동일 |
| Weight decay | 0.1 | 0 |
| Label smoothing | 0 | 0.1 |
| Warmup steps | 10 | 0 |
| Gradient clipping | 1.0 | 2.0 |
| Scheduler | linear | cosine |
| Seed | 3407, 3408, 3409 | 동일 |

공통 조건은 micro batch 4, accumulation 1, 최대 32 epoch, 최소 관찰 8 epoch, patience 6, min_delta 0.004다. 인코더·base revision·입력 형식·콜백을 유지한다. 데이터는 `hermes-ten-tools-v3`, 카탈로그는 `hermes-ten-tools-2289142ab57a`다. 원래 train600을 동일한 scenario grouped stratified 5-fold로 나누며 fold당 학습480/검증120이다. fold 배정 SHA는 `5e823b9408bf8c410005042315c362b8291aed9efd3e461cec7cc79fbac33e9b`다.

조건을 미리 고정했으므로 이번에는 TPE로 다음 조건을 선택하지 않는다. test20 점수는 기존 방식대로 최종 head의 참고 값만 남긴다. 반복 관찰한 test20으로 설정을 추가 선택하지 않는다. 비교 결론은 seed별 paired validation 차이와 변동을 확인한 뒤 낸다.

## 독립 test 준비 상태

| 항목 | 수량 |
|---|---:|
| 목표 | 도구별 20개 · 총 200개 |
| 이전 실제 요청 후보 | 13개 |
| 새 사용자 메시지 | 78개 |
| 새로 추가 가능한 후보 | 0개 |
| AI 검토자의 단일 도구 합의 | 1개 |
| 사람 검토 완료 | 0개 |
| 사람 검토 기준 부족량 | 200개 |

동일 프로필의 이전 2,050개 메시지 전체 fingerprint가 새 DB의 앞부분과 일치함을 확인한 뒤 2,128개 기록을 읽기 전용으로 조사했다. 새 메시지는 길이·도구 연관성·압축/활성 상태 등 기존 수집 기준을 통과하지 못했다. 정제 정책이나 필터를 완화해 수를 채우지 않았다.

기존 후보 13개는 최신 학습 자료를 포함한 2,124개 과거 레코드와 다시 비교했다. 검사 범위에서 정확·근접 중복과 출처/입력 경계 위반은 없었다. 후보 원문과 AI 제안은 private 검토 패킷에 보존하고, MLflow에는 집계 readiness JSON만 올린다. 사람 정답이 없으므로 동결·채점하지 않는다. 기존 감사 도구의 균형 판정은 100개 기준이며 200개 준비 완료 판정에 재사용하지 않았다.

## 코드와 검증

- [OOF 비교 스크립트](../../scripts/compare_laya_cv_errors.py): 저장 예측을 비교하며 모델을 재실행하지 않는다.
- [재검증 계획](../../configs/training/laya-ten-tools-seed-confirmation-cv5-20261024.json): 6개 조건과 5개 fold를 고정한다.
- [CV 실행기](../../scripts/run_laya_cv_experiment.py): 명시적 조건을 기존 fold supervisor에 전달한다.
- [prepare](../../scripts/prepare_laya_cuda_experiment.py): 일시적인 조회·업로드 오류를 최대 8회 재시도한다. Run 생성은 응답 유실 때 중복될 수 있어 자동 재시도 대상에서 제외했다.
- [수집기](../../scripts/collect_frozen_eval_candidates.py): 목표 수와 추가 중복 제외 자료를 CLI로 받는다.

서버의 분리된 임시 checkout에서 통합 테스트 **48개**, 수집 테스트 **4개** 통과. 최초 전체 실행에서 Node가 PATH에 없어 n8n 검사 1개가 실패했으나, 설치된 Node 경로를 지정한 재검증에서는 48개 모두 통과했다. OOF 맥락 필드 추가 후 해당 테스트 2개와 실제 600개 비교도 다시 통과했다. 기존 데이터 검증은 **130개 사례 · 10개 도구 · 65개 시나리오 PASS**다. 새 의존성은 없다.

## 재현과 파일 위치

서버 저장소는 `/home/persona-ai/git/Jery-cuda`다. 아래 실행은 동일 state의 완료 fold를 건너뛰며 재개한다. 같은 state의 supervisor가 실행 중이면 잠금으로 중복 실행을 거부한다.

```sh
/home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_cv_experiment.py \
  --plan configs/training/laya-ten-tools-seed-confirmation-cv5-20261024.json \
  --model-config configs/training/laya-ten-tools-seed-confirmation-cv5-20261024-model.json \
  --state-dir outputs/laya-ten-tools-seed-confirmation-cv5-20261024/tracking \
  --python /home/persona-ai/envs/jery-cuda/bin/python --detach

python3 scripts/compare_laya_cv_errors.py \
  --state-dir outputs/laya-ten-tools-fixedlr-params-cv5-20261022/tracking \
  --output outputs/laya-oof-review-new
```

서버의 OOF 결과는 `outputs/laya-oof-review-20261024/`, 학습 상태는 위 state-dir에 있다. Mac의 실제 요청 검토 패킷은 `data/local/frozen-eval-20261024/review-packet/`, 집계는 `outputs/frozen-eval-20261024/readiness.json`이다. 원문·키·가중치·실험 산출물은 Git에서 제외한다.

구현 커밋은 `6e4562a` · `dev-codex`이며 push 및 서버 반영을 완료했다. 학습 시작과 첫 epoch 확인 결과는 아래 실행 확인에 기록한다.

## 실행 확인 · 2026-10-24 14:06 KST

첫 조건 `baseline-seed3407`의 fold1에서 CUDA 학습과 **epoch3의 MLflow 지표 기록**을 확인했다. 학습 PID는 281752이며 GPU 사용량은 확인 시점 약 1.35GB였다. 다른 프로젝트의 GPU 프로세스는 변경하지 않았다.

MLflow에서 전체 Parent `af613b715d804393b17872be8d5ad671` → 조건 `7bbaebad27ad494b85e8b1464659edfc` → fold `f7a6bb202a48472b95a3446e55444996`의 연결과 실제 LR·seed·배치·optimizer 파라미터를 확인했다. 전체 30회 완료 보고는 아니며 supervisor가 이후 조건을 순차 실행한다.

Parent Artifact의 `analysis/oof-summary.json`, `analysis/oof-cases.json`, `analysis/frozen-eval-readiness.json` 업로드 후 GET/SHA 검증을 완료했다. 원문 실제 요청·가명 키는 업로드하지 않았다.
