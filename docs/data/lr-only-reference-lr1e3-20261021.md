# LR 외 전체 설정을 lr1e3 Parent 선택 Child로 통일 — 2026-10-21

사용자 요청으로 `laya-ten-tools-train600-lr1e3-20261021` Parent(6261a1bdca194867a835cbfb2216493f)의 실제 Child 설정을 기준으로 다음 LR-only CV 계획을 준비했다. 학습 중단 지시는 유지하며 새 MLflow run을 만들거나 학습을 재개하지 않았다.

## 기준 Child

Parent의 서버 state가 최종 선정한 `selected_child_run_id`는 **c04cdad0246b4d6896ef89e5f9032ce2**(두 번째 Child)다. Best-trained validation accuracy0.95이며 동률에서는 낮은 LR 기준이다. Test로 기준을 고르지 않았다. Parent의20개 Child를 모두 비교하니 LR 외에 WD와 label smoothing도 서로 달랐다. 따라서 Parent 전체의 공통값이라고 가정하지 않고 Parent가 실제 최종 선정한 Child의 configuration.json을 복사했다.

원본 Child configuration.json SHA256: `7f1dae65c578430c8bab449796052db832dc07757761c70ce2adacc56ee21fce`. 같은 SHA와 출처 run ID를 새 plan 및 로컬 reference-config.json에도 기록했다.

| 항목 | 중단한 CV 설정 | 새 기준값 |
|---|---:|---:|
| Warmup steps | 75 | 10 |
| Max gradient norm | 2.0 | 1.0 |
| Scheduler | cosine | linear |
| Weight decay | 0.1 | 0.05 |
| Label smoothing | 0 | 0.1 |
| Effective batch | 4 | 4 |
| Seed | 3407 | 3407 |
| 최대 epochs | 32 | 32 |

나머지 model revision/SHA, encoder 설정, 입력 길이, checkpoint, memory guard/fraction, eval batch, 균형 비율과 validation callback도 실제 기준 Child와 모두 동일하다. Callback은 최소8epoch/patience6/min_delta0.004 및 초기 head 보호다. Parent 기본 model-config의 label_smoothing0.05를 잘못 복사하지 않고 실제 Child의0.1을 따른다.

## 불변식과 상태 보존

새 계획/모델 설정:

- configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-20261021.json
- configs/training/laya-ten-tools-lr-only-cv5-lr1e3ref-model-20261021.json

LR 탐색1e-3~5e-3와 같은5-fold/seed/원본train600 pool은 이전 요청대로 유지한다. Hyperopt는20조건이며 WD/smoothing은 singleton으로 고정한다. 모델 설정에는 기준 Child의 전체 config snapshot을 reference_model_config로 기록한다. 실행 전 head_learning_rate 및 결과 저장용 models_dir/outputs_dir를 제외한 전체 key/value가 source snapshot과 같아야 한다. Artifact 경로는 이전 결과를 덮어쓰지 않도록 새 namespace를 쓴다. 모델의 기본 head LR은 범위 하한1e-3이며 실제 학습 LR은 후보값으로 덮어쓴다.

중단한 기존 CV state/원래 plan/config/weights/MLflow 기록은 변경하지 않는다. 새 설정을 서버에 배포해도 prepare·detach·resume 명령을 실행하지 않는다. 학습 프로세스가 없는 상태와 기존 state interrupted를 확인한다. 재학습은 별도 사용자 요청이 있어야 시작한다.

## 검증

원본 데이터 validator PASS(130examples/10tools/65isolated scenarios), CV 통합4개 PASS. 실제 원본 Child config와 비-LR 전체 값 equality 검증 및8개 실제 Hyperopt/TPE 후보의 불변식 검증 PASS. 회귀 테스트는 LR·새 결과 저장 경로 변경은 허용하되 warmup/smoothing뿐 아니라 eval_batch_size/gradient_checkpointing처럼 탐색 공간 밖 값의 drift도 거부한다. Graphify AST update 수행. 추가 의존성과 모델 코드 변경 없음.
