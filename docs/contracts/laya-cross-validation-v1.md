# Laya stratified grouped cross-validation tracking v1

검증 목적: 고정 10개 도구 routing에서 같은 fold로 학습 조건을 비교한다. 기본 모드는 LR만 탐색하며, 선택적 `fixed_lr_parameters` 모드는 LR을 고정하고 optimizer 조건을 탐색한다. 기존 model result-v2와 decision-export-v2 계약은 변경하지 않는다.

## 데이터

- Pool: hermes-ten-tools-v3 원래 train600. 원래 validation20은 CV에 사용하지 않고 test20은 train/validation fold에 포함하지 않는다.
- Label: gold route alias를 metadata.alias_to_tool로 실제 도구 ID에 연결한 값. T0 같은 위치 alias로 층화하지 않는다.
- Group: scenario_id. 파생 표현/맥락 형제는 같은 fold로 유지한다.
- 현재 pool의355개 group은 각각 하나의 label만 포함한다. 표준 라이브러리로 class별 group을 큰 것부터 가장 덜 찬 fold에 배정하고 seed3407로 동률을 결정한다. Algorithm ID는 homogeneous-scenario-stratified-largest-first-v1이다. 일반 mixed-label group은 조용히 분할하지 않고 거부한다.
- Five validation folds가 train600을 중복 없이 정확히 한 번씩 덮는다. 실제 fold마다 train480/validation120, 각 도구 train48/validation12다. Fold 내부 train/validation group과 source ID는 겹치지 않는다.
- 원래 test20 bytes와 원본 train row content를 보존한다. 각 fold에서 test20은 최종 선택 head를 한 번 평가할 때만 사용한다. Fold 분할, epoch 선택, 하이퍼파라미터 탐색 loss에 test를 사용하지 않는다. 반복 관찰한 합성 test의 독립 일반화 평가를 주장하지 않는다.
- folds.json에 source hash·fold별 validation case ID·분포·분할 SHA를 고정한다. 파생 manifest는 cross_validation metadata와 profile을 명시하고 원본 dataset 파일을 덮어쓰지 않는다.

## MLflow 구조와 지표

`CV Parent -> 조건 trial20개 -> fold training Child5개`의3단계다. 같은 조건 trial의5개 모델은 같은 초기 base, seed, optimizer 설정을 사용하며 다른 fold 입력만 받는다. 전체100회 학습이다. 각 fold 모델은480개로 학습되므로 full600 단일 모델의 평가로 표시하지 않는다. 별도 full600 재학습을 자동으로 추가하지 않는다.

Trial의 val_accuracy/val_macro_f1은5개 fold 최종 selected-serving head의 검증 평균, suffix _std는 sample standard deviation(ddof=1)이다. test_accuracy/test_macro_f1과 _std는 같은 고정 test20에 대한5개 fold-trained head의 평균/표준편차다. CV tuning의 선택 metric은 cv_search_accuracy = fold별 best_trained validation accuracy의 평균이다. Baseline fallback과 구분하기 위해 이 값은 val_accuracy와 다를 수 있다. TPE loss는1-cv_search_accuracy이며 test는 전달하지 않는다.

각 fold Child는 기존7개 metrics/실제 optimizer params/모델·JSON Artifact 계약을 따른다. Fold Child 표의 val은 마지막 epoch 값이며 선택 epoch의 값은 details/report.json을 따른다. Trial 부모에 기존 supervisor가 기록하는 best_val_accuracy/selected_test_accuracy는 최고 fold에 관한 legacy 진단값이며 primary CV metric으로 사용하지 않는다. CV trial은 jery.primary_metric=cv_search_accuracy 태그와 cv-summary.json을 제공한다.

전체 Parent의 summary.json은20개 trial의 Hyperopt document·평균 점수·fold run ID·선택 trial을 포함한다. Fold 데이터는 전체 Parent data/fold-N/에 저장하고 각 Child configuration.json의 data_parent_artifact_uri가 해당 fold를 가리킨다.

## 복구

CV supervisor와 기존 fold supervisor는 각각 lock을 갖는다. 재실행 시 plan/config/folds hash를 확인하고 완료 trial/fold를 재학습하지 않는다. 모델 report가 이미 있으면 기존 supervisor의 업로드 복구를 재사용한다. 실패 시 CV state와 MLflow Parent에 실패 상태를 표시한다.

## 선택적 기준 Child 전체 설정 고정

reference_model_config가 있는 CV plan은 head_learning_rate와 models_dir/outputs_dir를 제외한 모든 model config key/value가 snapshot과 일치해야 한다. 기준은 parent 기본값이 아닌 실제 Child configuration.json이다. LR 이외의 전처리·학습·평가·memory 설정 drift도 거부한다. 이 snapshot을 추가해도 과거 run/state/학습 결과를 다시 해석하거나 변경하지 않는다.

## 선택적 LR 고정 파라미터 탐색 · 2026-10-22 확장

기존 plan은 기본 LR 단독 동작을 유지한다. `cv.search_mode=fixed_lr_parameters`인 새 plan에서만 다음 규칙을 적용한다. 데이터·모델 입출력과 지표 정의는 바꾸지 않는다.

- `head_lr`의 양 끝은 기준 Child의 실제 LR과 정확히 같다. LR은 Hyperopt 확률 변수가 아닌 상수다.
- `first_trial_parameters`는 기준 설정과 일치해야 한다. 첫 Hyperopt document에는 이 조합을 넣어 이후 TPE 이력에도 반영한다.
- 후속 19개 조건은 weight decay·label smoothing·warmup·accumulation·clipping·scheduler를 탐색한다. 기준 조합 전체의 재추출은 제외한다.
- Root model config는 기준 snapshot과 일치하며 저장 경로만 분리한다. 조건별 config는 허용한 탐색 override를 반영하고, 한 조건의 5개 fold에는 같은 값을 전달한다.
- Seed·micro batch·에폭·콜백·입력 형식·고정 base는 유지한다. 실제 유효 배치는 micro batch × accumulation으로 기록한다.
- 기준 설정은 원래 run의 하이퍼파라미터를 뜻한다. 새 실험은 CV 분할을 사용하므로 원래 단일 split 점수와 같은 결과를 보장하지 않는다.

## 후보 중복 방지 · 2026-10-24 확장

새 후보는 저장된 모든 기존 파라미터 조합과 비교한다. 고정 LR의 유한 탐색에서 TPE 제안이 중복되면 미사용 조합을 seed로 선택하고 실제 선택을 Hyperopt document에도 기록한다. `proposal`은 `reference`, `tpe`, `unseen_grid`로 구분한다. 남은 조합이 없으면 명시적으로 종료한다. 기존 저장 후보는 재추출하지 않으며, 완료된 실험의 기록은 수정하지 않는다.

## 명시적 조건·seed 반복 · 2026-10-24 확장

`cv.search_mode=fixed_conditions`는 Hyperopt 대신 `trials`에 미리 선언한 조건을 순서대로 실행한다. 전체 Parent → 조건·seed Child → fold Child 구조와 기존 지표 정의를 유지한다. 현재 계획은 기준·최고 설정 × seed 3407·3408·3409, 총 6조건·30회 학습이다. seed3407 반복은 재현성 확인을 위한 의도적인 반복이며 탐색 중복 제거 대상이 아니다.

각 조건의 seed·micro_batch·grad_accum·optimizer 설정을 실제 fold config와 MLflow params에 같이 전달한다. 모든 조건은 동일한 고정 fold 파일을 사용하며 trial의 export_dir 변경과 fixed_conditions 위반을 거부한다. 요약의 validation 순위는 기술 통계이며 test로 다음 조건을 고르지 않는다. `parameters`와 `condition`을 state에 보존하고 가짜 Hyperopt document를 만들지 않는다.

prepare의 실험 조회·Artifact PUT/GET 검증은 기존 transient 오류 분류를 재사용해 최대 8회 재시도한다. HTTP 4xx(429 제외)·해시 불일치는 즉시 실패한다. 응답 유실 시 중복 Run을 만들 수 있는 `runs/create`는 자동 재시도하지 않는다. 이 경우 원격 생성 여부를 확인한 뒤 복구해야 한다.
