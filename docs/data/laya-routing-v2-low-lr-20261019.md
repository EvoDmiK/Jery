# Routing v2: 초기 LR 재탐색

이 LR 단독 plan은 준비만 한 뒤 사용자 추가 요청에 따라 학습 시작 전에 폐기했다. Parent `c755eeb007b5481a9cc5de444ab8a085`는 `KILLED`이며 Child가 없다. 후속 실행은 [LR·가중치 공동 탐색](laya-routing-v2-joint-search-20261019.md)을 따른다. 아래는 처음 준비한 설정의 기록이다.

사용자의 성능 저하 지적에 따라 기존 고정 LR Parent `bc61fce284ae468e8b7022508f36e819`를 중단했다. 완료된 가중치 1.0/0.5의 10에폭 결과는 보존하고, 진행 중이던 0.25 조건은 중단 상태로 보존한다. 다른 회사 GPU 작업은 유지한다.

기존 초기 LR은 `4.879746787617291e-5`였다. 가중치 1.0 조건의 최고 trained validation 정확도는 10.89%, 가중치 0.5는 12.50%로 baseline 16.13%에 못 미쳤다. 두 조건 모두 4에폭 이후 train loss가 하락하면서 validation soft CE는 상승했다. LR 고정은 가중치 비교를 위한 통제였지만, 별도 LR 재검증도 필요했다. LR이 유일한 원인이라고 확정하지 않는다.

## 변경 범위

`configs/training/laya-routing-v2-low-lr-20261019.json`은 기존 Hyperopt TPE 실행기를 재사용한다. 초기 head LR만 log-uniform `1e-6~1e-5`에서 5회 탐색한다. 기존보다 약 4.9~48.8배 낮은 범위다. weight decay 0.01, label smoothing 0, window_none loss 가중치 1.0, seed 3407, 유효 batch 4, warmup 64스텝과 linear scheduler를 유지한다. 가중치 1.0 대조군과 비교해 LR의 효과를 분리한다. 기존에도 linear 감소는 있었으며, 이번 변경은 초기 LR의 검색 범위를 낮추는 것이다.

각 Child는 같은 base에서 새로 시작해 전체 데이터를 10에폭 학습한다. 데이터는 `mcp-routing-v2` / `decision-export-v1`이며 train/validation/test 668/248/246개, 창 4,172/1,592/1,577개를 유지한다. 최고 trained validation 점수만 Hyperopt에 전달하고 test는 탐색에 사용하지 않는다. baseline 보호와 min_epochs 10, patience 5, min_delta 0.004는 유지한다. min_epochs와 예산이 모두 10이므로 이 실험의 조기 종료는 10에폭보다 앞서 적용되지 않는다.

MLflow `Jery`의 별도 Parent–Child를 사용한다. Child params 4개·metrics 7개를 유지하고 나머지는 JSON artifact로 저장한다. 모델은 serving/best-trained/last-trained로 구분한다. 의존성은 기존 `requirements-decision-cuda.txt`이며 코드·모델 아키텍처를 바꾸지 않는다. 결과는 합성 미검토 정답에 대한 탐색 실험이다.

## 재현

회사 서버 checkout `/home/persona-ai/git/Jery-cuda`에서 다음을 실행한다.

```sh
/home/persona-ai/envs/jery-cuda/bin/python scripts/prepare_laya_cuda_experiment.py \
  --plan configs/training/laya-routing-v2-low-lr-20261019.json \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-low-lr-20261019/tracking
TORCHINDUCTOR_COMPILE_THREADS=4 /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
  --python /home/persona-ai/envs/jery-cuda/bin/python \
  --model-config configs/model/laya-cuda-routing-v2-20261019.json \
  --state-dir outputs/laya-routing-v2-low-lr-20261019/tracking --detach
```

기존 학습 head를 이어 학습하지 않고, 새 Parent에서 초기 LR을 비교한다.
