# Laya routing v2: 데이터 보존과 가중 loss 제어 실험

기존 학습은 사용자 지시에 따라 중단하고 결과를 보존한다. 기존 Mac Parent는 `9325d1ca84944fe281f94b519c59f305`, CUDA Parent는 `29552487a8a94b10ad942d9bbc331953`다. 재학습은 회사 RTX 5070 Ti의 별도 CUDA Parent에서 시작한다.

## 데이터와 원인 진단

`mcp-routing-v2` 원본 668/248/246개와 frozen `decision-export-v1`의 4,172/1,592/1,577개 창을 그대로 사용한다. 카탈로그는 `hermes-mcp-57e701d87790`다. 그룹 분할·정답·후보 창을 재생성하거나 변경하지 않는다. 모든 정답은 합성 미검토 상태다.

validation 원본 248개 중 232개는 도구 선택 요청이지만, 창 1,592개 중 1,305개는 `window_none`이 정답이다. 무조건 `window_none`을 반환해도 창 정확도는 81.97%, 요청 정확도는 3.23%다. Mac 첫 trial은 창 정확도 8.79→78.96%, 정답 도구가 있는 창의 정확도 26.81→1.70%, 요청 정확도 16.13→4.84%로 변했다. 창 단위 loss 개선과 최종 라우팅 개선이 다르다는 증거다. 한국어가 주원인이라는 결론은 내리지 않는다.

## 첫 제어 실험

`configs/training/laya-routing-v2-controls-20261019.json`으로 다음 조건만 바꾼다.

| 조건 | `window_none` loss 가중치 | train 평균 가중치 |
|---|---:|---:|
| 대조군 | 1.0 | 1.0 |
| 가중치 0.5 | 0.5 | 0.8 |
| 가중치 0.25 | 0.25 | 0.7 |

각 조건은 10에폭이며, LR `4.879746787617291e-5`, weight decay `0.01`, label smoothing `0`, seed `3407`, micro batch 4 / accumulation 1을 고정한다. 매 조건은 같은 base 모델에서 새로 시작한다. sampled train 인덱스는 기존 40/40/10/10의 8,530개이며 원래 4,172개 창을 모두 포함한다.

정답이 `window_none`인 행에만 가중치를 곱하고 sampled train 전체의 평균 가중치로 정규화한다. sample 가중치·정답은 loss 계산에만 쓰며 입력 특징에 넣지 않는다. validation/test로 정규화하지 않는다. 이 실험에서 유효한 가중치를 아직 확인하지 않았으므로 성능 개선을 주장하지 않는다.

## 학습·선택 정책

[학습 결과 v2 계약](../contracts/laya-training-result-v2.md)을 따른다. 학습한 head끼리의 최고 validation 점수를 검색에 사용하고 baseline epoch 0으로 점수를 올림하지 않는다. 최종 serving head는 baseline과 비교해 보호한다. 조기 종료는 학습 모델의 유의미한 개선을 기준으로 판단하며 최초 10에폭 이후에만 bad epoch를 누적한다.

최종 요청 정확도와 정답 도구 창 성능뿐 아니라 `no_tool`·`clarify`의 성능, 불필요한 호출 비율, coverage도 함께 확인한다. test로 가중치나 하이퍼파라미터를 고르지 않는다. 이 3조건 제어 실험과 별도로 Hyperopt v2 목적함수도 trained-only로 수정하며, 이후 최대 64에폭 검색에 사용할 수 있다.

## 실행·검증

서버 checkout은 `/home/persona-ai/git/Jery-cuda`, 환경은 `/home/persona-ai/envs/jery-cuda`다. 설정 파일 `configs/model/laya-cuda-routing-v2-20261019.json`을 사용한다.

```sh
python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-routing-v2-controls-20261019.json \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-20261019/controls
TORCHINDUCTOR_COMPILE_THREADS=4 python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-20261019/controls --detach
```

실행 전 해당 전용 Python 환경을 사용해야 한다. params 4개·metrics 7개와 Parent–Child 구조를 유지하고 나머지는 JSON artifact로 저장한다. Serving, best-trained, last-trained 가중치를 구분해 업로드한다. 입력·weights·로그는 Git 밖에 둔다.

Claude는 weighted loss와 독립 학습 선택 정책, AGY는 독립 결과 감사, Codex는 실행기·데이터 계약·MLflow 통합을 담당한다. 새 학습 시작 전 단위 테스트와 작은 CUDA 전체 경로 검증을 수행하고, 실제 실행 상태와 결과는 해당 MLflow Parent 및 Obsidian 작업기록에 기록한다.

### CUDA 실행 경로 확인

각 split에서 기존 결정 유형별 사례 한 개의 모든 창을 가져온 작은 fixture로 10에폭·30스텝을 완료했다. `window_none` 가중치 0.25, train 평균 0.7이 적용됐고 최초 10에폭을 모두 관찰했다. 최고 학습 head(epoch 1)와 serving baseline(epoch 0)이 따로 저장됐으며, 재로딩 후 확률 차이 0 및 frozen encoder 불변을 확인했다. 이 fixture는 구현 검증용이며 전체 데이터 성능 근거가 아니다.

별도 출력 경로에서 native checkpoint 30을 읽어 optimizer state 30→31과 global step 30→31을 확인했다. 원본 checkpoint의 모든 파일 해시는 재개 전후 동일했다. 증거는 Git 밖 `outputs/laya-routing-v2-20261019/`와 새 Parent의 readiness JSON artifact에 보존한다.

Claude의 읽기 전용 통합 검토 후, prepared configuration과 실제 실행 configuration이 같고 policy version도 일치하는지 실행 전 검사하도록 보강했다. Parent의 `best_val_accuracy`는 v2에서 최고 trained validation 점수를 뜻한다. 최종 test는 별도로 보호된 serving head 기준이다.

### 전체 데이터 재학습 시작

2026-10-19 14:24 KST, 회사 RTX 5070 Ti에서 [새 Parent](https://mlflow.dove-nest.com/#/experiments/47/runs/bc61fce284ae468e8b7022508f36e819)를 시작했다. 첫 [Child](https://mlflow.dove-nest.com/#/experiments/47/runs/18dc59a3819d42739340f20890702b08)는 가중치 1.0 조건이며 0.5·0.25 조건이 순차 실행된다. 실제 optimizer step 128 이상과 유한한 loss·gradient를 확인했다. 에폭 곡선은 해당 에폭의 validation이 끝난 뒤 기록한다. Mac 학습은 중단 상태다.

학습 코드 기준은 `275fee1`이다. Claude 구현 `b3ea679`는 `2480f06`으로, AGY의 `1e5610f`·`08a7c3d`·`e7a7655` 최종 변경은 생성 JSON을 제외하고 `275fee1`로 통합했다. 통합 후 전체 테스트 132개, v1/v2 데이터 검증 및 실제 CUDA smoke 독립 감사를 통과했다. 데이터셋과 readiness 증거 JSON 다섯 개를 새 Parent에 업로드하고 재조회 해시를 검증했다. 진행 상태는 시점별 관측이며 최종 성능 개선 여부는 아직 평가하지 않았다.
