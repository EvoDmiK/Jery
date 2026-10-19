# Laya Routing v2 독립 감사 및 에폭 궤적 진단 보고서 (2026-10-19)

- **작성 브랜치**: `dev-agy`
- **감사 도구**: [`apps/review/routing_v2_audit.py`](../../apps/review/routing_v2_audit.py)
- **단위 테스트**: [`tests/review/test_routing_v2_audit.py`](../../tests/review/test_routing_v2_audit.py)
- **JSON 요약 산출물**: [`docs/review/laya_routing_v2_audit_summary.json`](./laya_routing_v2_audit_summary.json)
- **데이터 및 라벨 상태**: `mcp-routing-v2` (합성 탐색용 데이터셋, `needs_human_review`, `execution_verified: false` — 사람에 의해 검토된 정답이나 실제 실행 성공 성능으로 주장하지 않음)

---

## 1. 독립 검증 요약 (Independent Evidence Verification)

읽기 전용 통합 자산(`/Volumes/Dove-Nest-SSD/projects/Jery/outputs`)을 수정 없이 검증한 결과, 관찰된 수치와 `outputs/laya-diagnostics-20261019/objective-audit.json`의 기록이 정확히 일치함을 독립적으로 확인했다.

### 1.1 검증 분할 구조 및 상수 음성 창 대조군 (`always-window_none` Control)
- **검증 익스포트**: `outputs/laya-full-20261017/export/validation.jsonl` (`sha256: aa2c7175a6fb1564af7e992faa8aa64dce45418a9f9e32308a17e075815baf1b`)
- **사례(Case) 분포**: 총 **248건** (`select_tools: 232`, `unsupported: 8`, `no_tool: 4`, `clarify: 4`)
- **윈도우(Window) 분포**: 총 **1,592개** (`window_none: 1,305` [81.97%], `tool: 235` [14.76%], `no_tool: 26` [1.63%], `clarify: 26` [1.63%])
- **상수 `window_none` 예측 대조군**:
  - 모든 윈도우에서 항상 `window_none`만 선택하면, 1,592개 중 1,305개 음성 창을 맞혀 **전체 윈도우 정확도(Window Accuracy)는 81.97% (0.819724)**에 달한다.
  - 그러나 사례 집계(aggregation) 단계에서는 모든 사례가 `unsupported`로 귀결되어, 실제 정답이 `unsupported`인 8건만 맞히고 232건의 도구 선택 사례를 모두 틀리므로 **최종 사례 정확도(Case Accuracy)는 3.23% (8/248)**, **도구 선택 정확도(Tool Accuracy)는 0.00%**로 붕괴한다.
  - 따라서 **전체 윈도우 정확도의 상승을 모델 성능 개선으로 해석하면 안 되며**, 반드시 양성 도구 창 재현율(`positive_tool_window_recall`), 음성 창 정확도(`window_none_accuracy`), 최종 사례 정확도(`case_accuracy`)를 분리해 확인해야 한다.

### 1.2 Mac 균형 탐색(`laya-balanced-20261018`) 에폭 궤적 및 조기 종료 진단

| 시행 (Trial) | 학습률 (`head_lr`) | 상태 | Baseline (Ep 0) Case / ToolWin / Win | Ep 1 Case / ToolWin / Win | Best Trained Ep / Case Acc | Last Trained Ep / Case Acc | Serving Selected Ep / Case Acc | Hyperopt Loss (`1 - val`) |
|---|---:|---|---|---|---|---|---|---:|
| `trial1` | `4.87975e-5` | finished | `16.13%` / `26.81%` / `8.79%` | `4.84%` / `1.70%` / `78.96%` | Ep 3 / `7.66%` | Ep 5 / `5.24%` | Ep 0 (Baseline) / `16.13%` | `0.8387` |
| `trial2` | `4.91474e-5` | finished | `16.13%` / `26.81%` / `8.79%` | `3.63%` / `0.85%` / `80.15%` | Ep 4 / `11.29%` | Ep 5 / `11.29%` | Ep 0 (Baseline) / `16.13%` | `0.8387` |
| `trial3` | `1.09020e-5` | finished | `16.13%` / `26.81%` / `8.79%` | `14.92%` / `20.00%` / `28.14%` | Ep 1 / `14.92%` | Ep 5 / `4.84%` | Ep 0 (Baseline) / `16.13%` | `0.8387` |
| `trial4` | `6.65576e-4` | finished | `16.13%` / `26.81%` / `8.79%` | `3.23%` / `0.00%` / `80.03%` | Ep 5 / `15.73%` | Ep 5 / `15.73%` | Ep 0 (Baseline) / `16.13%` | `0.8387` |
| `trial5` | `2.94591e-4` | interrupted | `16.13%` / `26.81%` / `8.79%` | `4.03%` / `1.70%` / `79.71%` | Ep 2 / `14.92%` | Ep 5 / `11.69%` | 미완료 (`None`) | N/A |

- **Trial 1 거짓 윈도우 정확도 상승**: Epoch 0에서 Epoch 1로 가면서 전체 윈도우 정확도는 `8.79% -> 78.96%`로 급증했으나, 이는 `window_none` 정확도가 `3.98% -> 95.86%`로 치우친 결과다. 같은 시점에 양성 도구 창 재현율은 `26.81% -> 1.70%`로 붕괴했고, 사례 정확도는 `16.13% -> 4.84%`로 하락했다.
- **Trial 4 회복 중 조기 종료**: Epoch 1~5 동안 사례 정확도가 `3.23% -> 8.06% -> 8.87% -> 11.69% -> 15.73%`로 매 에폭 단조 회복(양성 도구 창 재현율도 `0.00% -> 4.68% -> 6.81% -> 10.21% -> 17.45%`로 상승)했으나, Epoch 0 baseline(`16.13%`) + `min_delta(0.004)`를 5에폭(`patience=5`) 내에 넘지 못했다는 이유로 Epoch 5에서 조기 종료되었다.
- **Hyperopt 목적함수 바닥 고정(Flooring)**: 완료된 Trial 1~4의 실제 최고 학습 검증 성능(`best_trained_score`)은 각각 `7.66%`, `11.29%`, `14.92%`, `15.73%`로 뚜렷이 달랐으나, 서빙 모델이 모두 Epoch 0 baseline(`16.13%`)으로 복원되면서 Hyperopt에 전달된 손실(`1 - final_validation`)이 네 시행 모두 `0.8387`로 동일하게 고정되었다.

---

## 2. `laya-routing-v2` 감사 계약 및 검증 규칙

[`apps/review/routing_v2_audit.py`](../../apps/review/routing_v2_audit.py)는 다음 계약을 독립 검증한다:

1. **네 가지 모델/에폭 상태의 명시적 분리**:
   - `baseline_epoch0`: 학습 전 `before_validation` 결과 (Epoch 0).
   - `best_trained_validation`: 실제 학습된 에폭(`epoch >= 1`) 중 최고 검증 사례 정확도를 기록한 모델 (`selection.best_trained_epoch`, `selection.best_trained_score`, `report.best_trained`).
   - `serving_selected`: 서빙용으로 최종 채택된 모델 (`selection.selected_epoch`, `selection.best_score`, `report.validation`). Baseline 유지 허용(`selected_epoch == 0` 시 `baseline_retained_for_serving = True` 명시).
   - `last_trained`: 마지막 완료 에폭 모델 (`selection.completed_epochs`, `report.last_trained`).
2. **v2 필수 설정 및 선택 계약 검증**:
   - `config.training_policy_version == "laya-routing-v2"`
   - `config.window_none_loss_weight` $\in (0, 1]$ (진단 조건 `1.0`, `0.5`, `0.25` 여부 표시)
   - `config.validation_callback`: `min_epochs == 10`, `patience == 5`, `min_delta == 0.004`
   - `selection.search_score == selection.best_trained_score == max(metrics.jsonl validation.case_accuracy)`
   - `report.best_trained` 및 `report.last_trained` 디스크립터(`artifact`, `artifact_sha256`, `epoch`, `validation_case_accuracy`, `format`)와 `selection`·`metrics.jsonl` 간 정합성 대조.
3. **과도한 도구 호출 방지 진단**:
   - `window_none` 가중치를 낮출 때 모델이 모든 사례에서 도구를 호출하는 편향으로 흐르지 않도록 `unnecessary_call_rate` 및 `no_tool`/`clarify` 결정별 precision/recall/F1을 함께 추출·감시한다.
4. **검증 전용 순위 산정 및 레거시 호환**:
   - 설정 간 순위는 오직 검증 분할의 `search_score`(레거시는 최고 학습 검증 정확도)로만 매기며, `test` 지표는 순위 산정에 절대 사용하지 않는다.
   - 기존 보고서는 수정하거나 v2로 재해석하지 않고 `contract_mode: "legacy"`로 명시해 지원한다.

---

## 3. 재현 명령 (Reproduction Commands)

```sh
# 1. 통합 자산 대상 독립 감사 실행 및 JSON 요약 생성
python3 apps/review/routing_v2_audit.py \
  --output-json docs/review/laya_routing_v2_audit_summary.json

# 2. 전체 리뷰 감사 단위 테스트 실행
python3 -m unittest discover -s tests/review -v

# 3. 데이터셋 검증 스크립트 실행
python3 scripts/validate_dataset.py
python3 scripts/validate_mcp_dataset.py
```
