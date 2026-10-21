# Laya stratified grouped cross-validation tracking v1

검증 목적: 고정10개 도구 routing에서 LR만 탐색하고 나머지 모델/optimizer 조건을 고정한다. 기존 model result-v2와 decision-export-v2 계약은 변경하지 않는다.

## 데이터

- Pool: hermes-ten-tools-v3 원래 train600. 원래 validation20은 CV에 사용하지 않고 test20은 train/validation fold에 포함하지 않는다.
- Label: gold route alias를 metadata.alias_to_tool로 실제 도구 ID에 연결한 값. T0 같은 위치 alias로 층화하지 않는다.
- Group: scenario_id. 파생 표현/맥락 형제는 같은 fold로 유지한다.
- 현재 pool의355개 group은 각각 하나의 label만 포함한다. 표준 라이브러리로 class별 group을 큰 것부터 가장 덜 찬 fold에 배정하고 seed3407로 동률을 결정한다. Algorithm ID는 homogeneous-scenario-stratified-largest-first-v1이다. 일반 mixed-label group은 조용히 분할하지 않고 거부한다.
- Five validation folds가 train600을 중복 없이 정확히 한 번씩 덮는다. 실제 fold마다 train480/validation120, 각 도구 train48/validation12다. Fold 내부 train/validation group과 source ID는 겹치지 않는다.
- 원래 test20 bytes와 원본 train row content를 보존한다. 각 fold에서 test20은 최종 선택 head를 한 번 평가할 때만 사용한다. Fold 분할, epoch 선택, LR 탐색 loss에 test를 사용하지 않는다. 반복 관찰한 합성 test의 독립 일반화 평가를 주장하지 않는다.
- folds.json에 source hash·fold별 validation case ID·분포·분할 SHA를 고정한다. 파생 manifest는 cross_validation metadata와 profile을 명시하고 원본 dataset 파일을 덮어쓰지 않는다.

## MLflow 구조와 지표

`CV Parent -> LR trial20개 -> fold training Child5개`의3단계다. 같은 LR trial의5개 모델은 같은 초기 base, seed, optimizer 설정을 사용하며 다른 fold 입력만 받는다. 전체100회 학습이다. 각 fold 모델은480개로 학습되므로 full600 단일 모델의 평가로 표시하지 않는다. 별도 full600 재학습을 자동으로 추가하지 않는다.

Trial의 val_accuracy/val_macro_f1은5개 fold 최종 selected-serving head의 검증 평균, suffix _std는 sample standard deviation(ddof=1)이다. test_accuracy/test_macro_f1과 _std는 같은 고정 test20에 대한5개 fold-trained head의 평균/표준편차다. CV tuning의 선택 metric은 cv_search_accuracy = fold별 best_trained validation accuracy의 평균이다. Baseline fallback과 구분하기 위해 이 값은 val_accuracy와 다를 수 있다. TPE loss는1-cv_search_accuracy이며 test는 전달하지 않는다.

각 fold Child는 기존7개 metrics/실제 optimizer params/모델·JSON Artifact 계약을 따른다. Fold Child 표의 val은 마지막 epoch 값이며 선택 epoch의 값은 details/report.json을 따른다. Trial 부모에 기존 supervisor가 기록하는 best_val_accuracy/selected_test_accuracy는 최고 fold에 관한 legacy 진단값이며 primary CV metric으로 사용하지 않는다. CV trial은 jery.primary_metric=cv_search_accuracy 태그와 cv-summary.json을 제공한다.

전체 Parent의 summary.json은20개 trial의 Hyperopt document·평균 점수·fold run ID·선택 trial을 포함한다. Fold 데이터는 전체 Parent data/fold-N/에 저장하고 각 Child configuration.json의 data_parent_artifact_uri가 해당 fold를 가리킨다.

## 복구

CV supervisor와 기존 fold supervisor는 각각 lock을 갖는다. 재실행 시 plan/config/folds hash를 확인하고 완료 trial/fold를 재학습하지 않는다. 모델 report가 이미 있으면 기존 supervisor의 업로드 복구를 재사용한다. 실패 시 CV state와 MLflow Parent에 실패 상태를 표시한다.

## 선택적 기준 Child 전체 설정 고정

reference_model_config가 있는 CV plan은 head_learning_rate와 models_dir/outputs_dir를 제외한 모든 model config key/value가 snapshot과 일치해야 한다. 기준은 parent 기본값이 아닌 실제 Child configuration.json이다. LR 이외의 전처리·학습·평가·memory 설정 drift도 거부한다. 이 snapshot을 추가해도 과거 run/state/학습 결과를 다시 해석하거나 변경하지 않는다.
