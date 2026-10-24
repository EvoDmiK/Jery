# 설정 검증 정리와 성능 개선 순서

**3개 seed × 2개 설정 × 5-fold, 총 30회가 완료됐다.** 최고 설정은 모든 seed에서 validation F1이 개선됐지만, 기존 test20에서는 일관된 개선이 없었다. 다음 실험은 오류 경계의 데이터 품질과 도구 설명을 우선 확인한다.

## 완료 결과

아래 값은 각 seed의 5-fold 평균이다. [MLflow 실험](https://mlflow.dove-nest.com/#/experiments/47/runs/af613b715d804393b17872be8d5ad671)에서 원본 결과를 확인할 수 있다.

| Seed | 기준 Val F1 | 최고 Val F1 | 변화 | 기준 Test F1 | 최고 Test F1 |
|---|---:|---:|---:|---:|---:|
| 3407 | 0.7443 | 0.7643 | +0.0200 | 0.6453 | 0.6453 |
| 3408 | 0.7268 | 0.7496 | +0.0228 | 0.6374 | 0.6573 |
| 3409 | 0.7314 | 0.7608 | +0.0294 | 0.6307 | 0.5893 |
| 평균 | **0.7342** | **0.7582** | **+0.0241** | **0.6378** | **0.6306** |

평균 validation 정확도는 73.83%→76.17%다. 최고 설정은 validation 기준 후속 비교의 기준점으로 사용할 만하다. 다만 기존 test는 20개이고 반복 관찰했으므로 실제 요청 성능이나 통계적 유의성을 주장하지 않는다. 동일 자료·서로 겹치는 학습 fold에서 나온 15개 점수를 독립 표본 15개로 취급하지 않는다.

선택된 checkpoint의 train 정확도는 기준 설정에서 seed별 90.42%·94.04%·92.38%, 최고 설정에서 95.17%·90.04%·88.75%였다. 대응 validation 정확도는 73.17~76.83%다. 일반화 격차가 있으므로 epoch나 LR을 더 높이는 것을 우선하지 않는다. 이것만으로 한국어 표현력이나 특정 구조가 원인이라고 단정할 수는 없다.

## 이번 리팩토링

[prepare_laya_cuda_experiment.py](../../scripts/prepare_laya_cuda_experiment.py)의 Hyperopt 탐색·명시적 trial·LR 범위가 같은 `valid_parameter` 검사를 사용하도록 정리했다. 정수 조건, bool/NaN/무한대 거부, smoothing/weight 범위와 scheduler 허용값을 한 곳에서 관리한다. 상위에서 이미 거부하는 혼합 schedule 재검사도 제거했다.

제품 코드 33줄을 지우고 24줄을 추가했다. 새 의존성이나 실행 프레임워크 없이 입력 검증 중복만 줄였다. 학습 수식·설정·모델 가중치를 변경하지 않았으므로 이번 정리 자체가 정확도를 높였다는 뜻은 아니다. 완료된 실험의 기록도 그대로 유지한다.

서버의 분리된 checkout에서 계획 검증 9개·CV 7개·tracking 15개, **31개 테스트가 통과**했다. 7개 optimizer 항목의 정상·경계·잘못된 값을 탐색과 고정 trial 양쪽에 넣어 같은 판단을 확인했다. 실제 Hyperopt 경로도 검사했다. 기존 데이터 검증은 130개 사례·10개 도구·65개 시나리오 PASS, `git diff --check`도 통과했다.

```sh
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_cuda_experiment_plan.py
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_cv_experiment.py
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_full_tracking.py
python3 scripts/validate_dataset.py
```

## 성능 개선 우선순위

| 순서 | 제안 | 판단 기준 |
|---|---|---|
| 1 | 공통 오답 104개의 요청·맥락·정답 검토 | 문맥 부족, 라벨 문제, 명세와의 불일치 구분 |
| 2 | 혼동 도구 쌍의 독립적인 대조 사례 보강 | vault/edit, portfolio/order, history/current-price에서 오류 감소 |
| 3 | 후보 설명의 구분 기준 개선 | 데이터 고정 상태에서 설명만 바꿔 paired CV 비교 |
| 4 | 데이터 개선 후 encoder LoRA 소규모 비교 | 같은 검증 설계에서 head-only 대비 재현되는 이득 확인 |

**데이터:** 현재 10개 도구는 각각 60개로 수량이 균형을 이룬다. 추가 증강은 혼동되는 의미 경계에 집중한다. 예를 들어 같은 문서에 대해 ‘내용을 찾아 보여줘’와 ‘그 문장을 수정해줘’, 같은 계좌에 대해 ‘현재 보유량을 보여줘’와 ‘10주 매수해줘’를 구분하는 새 사례를 준비한다. 우선 3개 경계에 도구 양쪽 20개씩, 총 120개를 소규모 제안으로 잡을 수 있다. 기존 validation 문장을 바꿔 train에 넣지 않으며, 독립 시나리오로 만들고 표현 변형·세션 가족은 같은 분할에 둔다. 합성 보강은 합성으로 표시하고 사람 정답으로 승격하지 않는다.

**후보 설명:** ‘읽기·검색’ 대 ‘수정’, ‘조회’ 대 ‘주문 실행’, ‘기간별 과거 가격’ 대 ‘한 종목의 현재가’를 명시한다. 설명을 늘리기 전에 실제 원래 명세와 일치하는지 확인한다. 기존 비교 대상 10개 fold의 입력 잘림은 0건이어서 max_seq_length 확대를 우선할 근거는 없다. 설명 변경 시 카탈로그·export 버전과 해시를 갱신해 이전 결과와 구분한다.

**LoRA:** 현재 CUDA 경로는 head/scorer/type_emb만 학습하고 encoder를 고정한다. LoRA는 원래 가중치를 고정한 채 작은 저랭크 가중치를 학습하는 방법이다. [Hugging Face 공식 문서](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)를 참고하되, Laya에서의 호환성이나 성능 이득은 아직 검증하지 않았다. 후보 rank 4·8 정도의 작은 비교부터 검토할 수 있다. 기존 코드에는 trainable parameter 수와 encoder 불변성 검사, head 전용 저장·복원 계약이 있으므로 단순 config 추가만으로 실행되지 않는다. adapter 저장·복원과 새 학습 모드의 검증을 먼저 구현해야 한다.

## 다음 실험의 범위

최고 설정을 고정한 비교 기준으로 두고 **기존 데이터 / 대조 데이터 추가 / 후보 설명만 변경**을 각각 비교하는 것이 우선이다. LR·scheduler까지 동시에 바꾸면 어느 변경이 효과가 있었는지 판단하기 어렵다. 이 문서의 데이터 120개와 LoRA는 후속 제안이며 이번에 생성하거나 학습하지 않았다.

현재 fold와 OOF 오류는 이미 개발 판단에 사용됐으므로 후속 점수 역시 개발 검증 점수다. 최종 설정을 정한 뒤 독립 자료로 평가해야 한다. 실제 요청 test 준비는 후보 13개·사람 검토 완료 0개로, 목표 200개는 아직 미달이다. 최종 평가용 신규 자료를 확보하고 잠근 뒤 점수를 한 번 확인하는 절차가 남아 있다.
