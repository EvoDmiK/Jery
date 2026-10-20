# Laya 10개 도구 학습 가능성 검증 (2026-10-20)

`configs/model/laya-ten-tools-20261020.json`은 CUDA `laya-routing-v2`를 상속하고
`evaluation_task=ten_tool_choice`를 명시할 때만 새 경로를 쓴다. 이 키가 없으면 기존
구성·데이터·지표는 그대로다(`max_window_tools`만 단독으로 두면 거부한다).

## 입력 계약

- `load_export`는 `decision-export-v1` 또는 manifest 최상위에
  `export_version=decision-export-v2`, `export_scope=fixed_ten_tools_select_only`가 있는 export만 받고,
  CUDA는 manifest `evaluation_task=ten_tool_choice`도 확인한다.
  v2 export를 `evaluation_task` 없이 쓰면 거부한다.
- 각 행: source_id당 창 1개(`window_count=1`), `criteria`는 alias `T0..T9` 10개뿐이며
  `no_tool`/`clarify`/`window_none` 없음, `alias_to_tool` 순서 = criteria 순서,
  `original_decision=select_tools`, `acceptable_choices`의 서로 다른 tool은 정확히 1개(같은 도구의 action
  대안은 허용), gold 질량은 그 도구 alias 하나. state의 후보 카드는 요구하지 않는다(단일 창 경로).
  metadata의 blind 키(`source_id`, `window_index`, `window_count`, `export_id`, `window_tools`,
  `alias_to_tool`, `original_candidate_tools`)는 v1과 같다.
- `group_cases(rows, max_window_tools)` 기본값은 8이고 CUDA는 이 task에서만 10을 넘긴다.
  창이 하나이므로 집계는 항상 `path=single`이다.
- train 비율 `tool=1`, null 버킷 `0`은 해당 버킷의 원본 창이 실제로 0개일 때만 허용한다.
  null 창이 하나라도 있으면 버리지 않고 오류를 낸다.

## 지표 형태

`validation`, `test`, 에폭별 `train_evaluation`에 기존 키를 유지한 채 다음을 더한다.

```json
{"tool_macro_f1": 0.42, "macro_f1_scope": "tools",
 "case": {"per_tool": {"mcp__s__t0": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 4}}}}
```

`tool_macro_f1`은 gold support가 있는 도구별 F1의 평균이다(잘못 고르면 고른 도구의 FP,
정답 도구의 FN). `decision_macro_f1`은 이 task에서 자명하게 1.0이므로, root는
`macro_f1_scope == "tools"`일 때만 `val_macro_f1`/`test_macro_f1`을 `tool_macro_f1`로 대응한다.

## 학습 적합 진단

`evaluate_train_each_epoch=true`이면 매 에폭 oversampling 이전의 고유 train 행 전체를 평가해
`metrics.jsonl` 레코드의 `train_evaluation`에 기록하고, 보고서에
`train_fit={last_trained_epoch, last_trained_accuracy, best_accuracy, best_epoch, used_for_selection: false}`를
남긴다. checkpoint 선택은 여전히 validation case accuracy만 쓴다. `min_epochs=epochs=64`이므로
patience는 발동하지 않으며, 64 에폭을 마친 checkpoint의 재개는 optimizer step을 더하지 않는다.

진행 순서: `memorization` profile(train 20 = 도구당 2개, validation/test 각 20)로 train fit ≥ 95%를
먼저 확인하고, 통과한 뒤에만 `full` profile(train 60, 동일 validation/test 20)로 넘어간다.
두 root export는 모두 이 계약 검사·`group_cases(…, 10)`·`balanced_indices`를 통과함을 확인했다(학습 미실행).

```sh
PYTHONPATH=src python3 -m unittest discover -s tests/model -p "test_laya*.py"
```
