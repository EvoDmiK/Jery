# Routing v2: LR·window_none 가중치 공동 탐색

사용자 요청에 따라 초기 head LR과 `window_none_loss_weight`를 Hyperopt TPE에서 함께 탐색한다. LR 단독 실험은 준비 후 학습 시작 전에 폐기했다. 이전 고정 LR Parent `bc61fce284ae468e8b7022508f36e819`의 완료 결과·체크포인트는 보존했다.

## 실행 계약

- 초기 LR: log-uniform `1e-6~1e-5`.
- `window_none_loss_weight`: choice `[0.25, 0.5, 1.0]`.
- 5개 trial, 각 10에폭, 동일 base에서 시작, seed 3407·유효 batch 4·weight decay 0.01·label smoothing 0·warmup 64 및 linear scheduler 유지.
- 최고 trained validation 정확도만 검색에 사용한다. test는 검색하지 않는다. baseline 보호·min_epochs 10·patience 5·min_delta 0.004는 유지한다.
- MLflow Child params 5개: `head_lr`, `window_none_loss_weight`, `epochs`, `seed`, `effective_batch_size`. metrics는 기존 7개다. 나머지는 JSON artifact, 모델은 safetensors로 저장한다. 기존 params 4개 plan과 호환된다.

설정은 `configs/training/laya-routing-v2-joint-search-20261019.json`, 모델 설정은 기존 `configs/model/laya-cuda-routing-v2-20261019.json`이다. 데이터·모델 구조·CUDA 의존성은 바꾸지 않는다. 결과는 합성 미검토 데이터에 대한 탐색 실험이다.

## 데이터 형식 대조

[Unsloth 공식 데이터 가이드](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth#dataset-guide)는 `state`, `questions`, `gold`를 사용하며 `choice`의 정답은 `criteria`의 선택지 키에 대응한다. 설치된 native `FastDecisionModel.build_dataset`는 dict형 gold의 확률 분포도 처리한다. Jery 원본 JSONL을 직접 입력하지 않고 `decision-export-v1`으로 변환한다.

| Jery export | native 학습에서의 의미 |
|---|---|
| `state` | 요청·호출 전 맥락·현재 후보 카드 |
| `questions.route_in_window` | choice 지침과 선택지 criteria |
| `gold.route_in_window` | label과 허용 선택지 확률 분포 |
| `metadata` | 계보·원본 정답 감사용, 입력 특징에 넣지 않음 |

형식 호환과 원래 사전학습 데이터의 분포·품질이 같다는 주장은 별개다. 현재 criteria의 도구 설명은 `도구 T0` 같은 별칭이고 의미 정보는 state 카드에 있다. 설명은 120자로 압축하며, 원래 요청의 여러 후보 창을 별도로 학습하고 최종 집계한다. 이 표현·라벨 분포·집계가 적절한지는 스키마 통과만으로 증명되지 않는다. Laya의 원 사전학습 코퍼스와 동일하다고 주장하지 않는다.

`scripts/audit_laya_export_format.py`는 전체 7,341개 창을 native builder로 읽고 다음을 확인한다. 가중치를 로드하거나 GPU forward를 실행하지 않으며 파일을 덮어쓰지 않는다.

1. 모든 행 수용 여부와 native target의 선택지 인덱스·확률 정합성.
2. metadata의 원래 결정·허용 도구와 export 정답 대응.
3. native 학습 특징과 gold·metadata를 읽지 않는 추론 특징의 일치.
4. 실제 tokenizer에서 state·지침·선택지 잘림 및 선택지 충돌 여부.

회사 CUDA 환경의 Unsloth 2026.10.6 / Laya 0.4.9 / Transformers 5.17.0으로 실행한 결과 train 4,172·validation 1,592·test 1,577개 창을 모두 수용했다. 각 split의 skip·target 불일치·특징 불일치·잘림·충돌은 모두 0이다. 최대 sequence 길이는 각각 950/953/930토큰이며 설정 한도는 1,024, head 한도는 256이다. 원본 v1/v2 검증과 공동 탐색 통합 후 전체 테스트 134개도 통과했다. 상세 보고서는 Git 밖 native-format-audit.json과 MLflow JSON artifact에 보존한다.

## 재현

회사 서버 checkout `/home/persona-ai/git/Jery-cuda`, Python `/home/persona-ai/envs/jery-cuda/bin/python`, 의존성 `requirements-decision-cuda.txt`를 사용한다.

```sh
PYTHONPATH=src python scripts/audit_laya_export_format.py \
  --config configs/model/laya-cuda-routing-v2-20261019.json \
  --output outputs/laya-routing-v2-joint-search-20261019/native-format-audit.json
python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-routing-v2-joint-search-20261019.json \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-joint-search-20261019/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-joint-search-20261019/tracking --detach
```
