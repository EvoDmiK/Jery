# decision-export-v2: 고정 10개 도구 단일 선택

이 버전은 `export_scope=fixed_ten_tools_select_only`, `evaluation_task=ten_tool_choice`인 축소 진단 과제다. 원본 `mcp-routing-v2`와 기존 `decision-export-v1`의 의미·파일을 바꾸지 않는다. 파생 데이터 버전은 `hermes-ten-tools-v1`이다.

## 입력과 정답

행은 기존 adapter와 같은 `state`, `questions`, `gold`, `metadata` 구조다. `questions.route_in_window`는 `choice`이며 선택지 정확히 10개를 갖는다. 선택지는 매 행 독립적으로 섞은 `T0`~`T9` alias와 실제 기능의 짧은 한국어 설명이다. `no_tool`, `clarify`, `window_none` 선택지는 없다. 도구의 명세는 카탈로그에서 확인하며 실제 도구를 실행하지 않는다.

`state`에는 원본의 request와 호출 전 context만 넣는다. `gold.route_in_window`에는 제공된 10개 중 하나의 alias label과 확률 1을 넣는다. 원본 허용 대안 중 선택한 10개 도구에 속한 서로 다른 tool ID가 정확히 하나인 select_tools 사례만 사용한다. action은 예측하지 않고 metadata에 보존한다.

## 계보와 분할

각 원본 사례는 창 하나다. `window_index=0`, `window_count=1`이며 alias map과 window_tools가 정확히 10개다. adapter의 `original_candidate_tools`는 이 새 과제의 고정 후보 C를 뜻한다. 원래 원본 행의 후보 C와 허용 정답 전체는 별도 `source_candidate_tools`, `source_acceptable_choices`에 보존한다. `acceptable_choices`는 새 후보 내 유일한 정답 도구의 원본 action 대안들이다.

metadata에는 source_id, scenario_id, source_split, subset 카탈로그 버전, export 버전·scope·evaluation_task와 검토 상태를 남긴다. metadata와 gold는 입력 특징이 아니다. 기존 train/validation/test의 시나리오 그룹 경계를 유지하며, 같은 그룹을 서로 다른 split에 넣지 않는다. 고정 후보를 구성하며 추가된 원본 C 밖 도구가 있을 수 있으므로 기존 후보의 단순 부분집합 과제라고 주장하지 않는다.

두 profile을 별도 출력한다. request+context가 동일한 사례를 먼저 중복 제거한다. `memorization`은 train 도구별 2개(총 20개), `full`은 train 도구별 6개(총 60개)다. 두 profile의 validation/test는 각각 도구별 2개(총 20개)이며 동일한 기존 split 사례·선택지 순서를 사용한다. 각 클래스에서 서로 다른 시나리오를 먼저 선택한다. 기존 합성 정답은 미검토 상태이며 새 사람이 검토한 정답으로 승격하지 않는다. test는 이미 관찰된 원본의 회귀 평가이며 새로운 독립 성능 증거가 아니다.

## 모델과 지표

`max_window_tools=10`은 이 명시적 scope에서만 사용한다. 기존 모델의 기본 창 한도 8은 유지한다. train bucket ratios는 `tool:1, no_tool:0, clarify:0, window_none:0`이며 빈 null bucket만 0으로 허용한다. nonempty bucket에 0을 주어 사례를 버리면 안 된다. 이 과제에는 window_none loss 가중치 효과가 없고 1로 고정한다.

한 번의 선택 결과를 alias map으로 원 tool ID에 대응시킨다. 창 분할·다수결·재대결은 없다. 후보 밖 선택은 거부한다. 정확도와 macro F1은 도구 10개를 기준으로 계산한다. report의 기존 decision 지표는 유지하되 `macro_f1_scope=tools`, `tool_macro_f1`을 명시하고, 이 scope에서만 MLflow `val_macro_f1`, `test_macro_f1`을 tool macro F1에 연결한다. 지표 정의는 새 Parent에 JSON으로 남긴다.

각 epoch에서 학습 데이터도 평가해 epoch JSON에 `train_evaluation`을 남기고 report의 `train_fit`에 마지막 학습 head의 정확도와 최고 train 정확도/epoch를 기록한다. train 성능은 학습 경로 진단용이며 checkpoint 선택·Hyperopt 점수에는 쓰지 않는다. 검증 기준 trained 선택·baseline 보호, test 최종 1회 정책은 유지한다. 첫 memorization 실행은 최대 64에폭 동안 train 정확도 95% 이상을 확인하는 진단이며, 실패하면 더 큰 탐색을 자동으로 확장하지 않는다.
