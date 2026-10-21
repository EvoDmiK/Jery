# LR-only20회와 stratified grouped5-fold — 2026-10-21

사용자가 비-LR 조건을 고정하고 LR1e-3~5e-3만 탐색하되 검증을 Stratified K-Fold로 요청했다. 직전 공동 탐색 Parent3b7fd13fe2f74146a520fa64d0c2305d는20회 전부 FINISHED임을 확인했다.

## 고정 조건과 탐색

직전 공동 탐색에서 best-trained validation accuracy가 가장 높은 run c0635c4ce7c44e97aad5e9f695516a0b(0.85)를 기준으로 비-LR 조건을 고정했다. Test는 이 선택에 사용하지 않았다.

- LR만 log-uniform1e-3~5e-3, Hyperopt TPE20/startup6.
- seed3407, micro batch4/accumulation1(유효4), warmup75 optimizer steps, max_grad_norm2.0, cosine scheduler, WD0.1, label smoothing0.0.
- 최대32epoch, 최소8epoch 관찰/patience6/min_delta0.004, 초기 head 보호·best-trained 별도 저장 유지.
- 매 fold 동일 pinned Laya base에서 새로 시작한다. Encoder 고정/FP32 head·BF16 autocast, base SHA·입력 길이1024/256·CUDA memory fraction0.3 유지.

## 검증과 데이터

[CV 계약 v1](../contracts/laya-cross-validation-v1.md)을 따른다. 실제 도구 ID로 층화하고 scenario_id group을 보존한다. 현재 group은 모두 단일 label이므로 표준 라이브러리의 deterministic largest-first 배정으로 정확한5-fold 균형을 확보했다. 새 scikit-learn 의존성을 추가하지 않았다. [공식 stratified/grouped CV 설명](https://scikit-learn.org/stable/modules/cross_validation.html)의 class 비율 보존과 group 비중첩 원칙을 적용하되 scikit-learn 클래스를 호출했다고 주장하지 않는다.

원래 train600만 CV pool이다. 각 fold480train/120validation, 도구당48/12.355개 scenario group 누수0, 전체600개 validation coverage1회, original test20 byte 보존. Original validation20은 CV에 사용하지 않는다. 모든20 LR trial이 같은 고정5-fold를 사용한다. 총100회 모델 학습으로 기존 단일 split20회보다 학습량이 늘어난다.

Test20은 각 fold head 최종 평가에만 쓰고 TPE/epoch 선택에는 쓰지 않는다. Trial test값은5개 fold-trained 모델 결과의 평균이며 full600 단일 모델 성능이 아니다. Trial val 평균·sample std와 각 fold별 지표/모델을 MLflow3단계 구조로 남긴다. Primary selection은 cv_search_accuracy(5개 fold best-trained validation accuracy 평균)다. 상세 metrics 의미는 계약을 따른다.

## 구현과 검증

Fold builder `scripts/build_laya_stratified_folds.py`, CV supervisor `scripts/run_laya_cv_experiment.py`를 추가했다. 기존 full supervisor는 fixed trial별 export_dir를 받아 같은 LR의5개 fold를 학습하며 각 export manifest/SHA를 검사한다. 기존 prepare·CLI·모델 학습/업로드/복구를 재사용하고 모델 코드는 변경하지 않았다. Fold별 데이터 Artifact 참조도 실제 사용한 fold로 연결한다.

테스트는 실제tool alias 복원·층화 균형·group/ID 누수 방지·전체 OOF coverage·heldout bytes·원본 보존·출력 덮어쓰기 거부·설정 drift 거부·LR-only 계획·test와 TPE 분리·학습 export override를 검증한다. 전체 테스트181개 PASS, LR-only8개 실제 Hyperopt 후보의 비-LR 불변식 검증 PASS, 기존 데이터 validator PASS. Graphify AST update 수행. 추가 의존성 없음.

## 실행

```sh
python3 scripts/build_laya_stratified_folds.py \
 --source outputs/laya-ten-tools-training-v3-final-20261020/augmented \
 --output outputs/laya-ten-tools-stratified5-20261021 --n-splits 5 --seed 3407
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_cv_experiment.py \
 --plan configs/training/laya-ten-tools-train600-lr-only-cv5-20261021.json \
 --model-config configs/training/laya-ten-tools-train600-lr-only-cv5-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-lr-only-cv5-20261021/tracking \
 --python /home/persona-ai/envs/jery-cuda/bin/python --detach
```

이미 생성된 fold 출력은 덮어쓰지 않는다. 재개는 기존 state와 같은 plan/config/fold 파일로 CV supervisor만 실행하며 process/lock을 확인한다. 학습 시작과100회 완료는 다른 상태이고 실제 상태는 tracking/state.json과 MLflow를 따른다.
