# Laya 전체 창 학습·평가 경로 (2026-10-17)

`mcp-routing-v2`의 모든 사례(train 668 / validation 248 / test 246)를 원래 후보 C 전체의 8도구 창으로 나눈 root export로 Laya head-only 학습과 case 단위 평가를 수행하는 CLI다. 이 작업에서는 경로 구현, 단위 테스트, 추론 전용 smoke, **8 step 속도 probe 1회**와 아래 tiny end-to-end 검증을 실행했다. 본 학습(3 child × 64 epoch)은 Codex의 supervisor가 순차 실행한다. 미검토 합성 데이터이므로 결과를 실제 사용 정확도로 해석하지 않는다.

## 합의된 계획 (coordinator 승인 `msg_02f8bf9a93fb`, `msg_59e5ed281a48`)

- 데이터: root `outputs/laya-full-20261017/export`(읽기 전용, manifest SHA-256 검증). train 4,172 / validation 1,592 / test 1,577 창이다. 모든 창을 재표집 없이 학습한다. train 창 레이블은 window_none 3,409, 도구 607, no_tool 78, clarify 78이다.
- 학습: frozen encoder, head·scorer·type_emb만 학습(기존 Unsloth MLX `DecisionTrainer`). microbatch 1 × accum 4 = 1,043 optimizer step/epoch, **64 epoch 고정**, 마지막 epoch head가 최종이다(epoch 선택 없음). child는 head LR 1e-4 / 3e-4 / 1e-3이고 seed는 3407이다([children](../../configs/model/laya-full-20261017-children.json)). parent의 child 선정은 최종 validation `case_accuracy`로 하고, 동률이면 낮은 head LR을 고른다(supervisor 담당). test는 선정에 쓰지 않으며, child마다 마지막에 1회 평가한다.
- 토큰 예산: 실행 시작 시 모든 split의 모든 창을 실제 Laya tokenizer와 `laya.common.build_sequence`로 검사한다. 옵션·지침·state 중 하나라도 잘리면 학습 전에 중단하고 `token_budget.json`에 창별 내역을 남긴다.

### label-blind case 집계

집계 함수는 `state`, `questions`와 계보 메타데이터(`source_id`, `window_index`, `window_count`, `export_id`, `window_tools`, `alias_to_tool`, `original_candidate_tools`)만 받는다. gold와 원본 decision·acceptable은 받지 않으며, 단위 테스트가 이를 확인한다.

- R0: case의 창 index가 0..count-1로 빠짐없이 있고, 창 도구들이 C를 정확히 한 번씩 덮는지 검사한다. 어긋나면 오류로 처리한다.
- R1: 모든 창을 1회씩 점수화하고 창별 top-1만 사용한다. 창 사이의 확률은 비교하지 않는다. alias는 즉시 원래 도구 ID로 복원한다.
- R2 (창 1개): top-1이 최종이다. 도구 → select_tools, no_tool/clarify → 그대로, window_none → unsupported(C 전체를 봤으므로).
- R3 (창 여러 개, 도구 승자 있음): 승자를 C 순서로 모은다. 각 승자의 원래 카드로 8개 이하의 새 창을 만들고 alias를 다시 붙여 라운드를 반복한다. chunk가 no_tool/clarify/window_none을 내면 그 chunk는 승자가 없는 것이다. 한 라운드에서 승자가 0이면 delegate다. 마지막 단일 창의 결과: 도구 → select_tools, no_tool/clarify → 그대로, window_none → delegate.
- R4 (창 여러 개, 도구 승자 없음): 창 표 {no_tool, clarify, window_none}의 다수결이다. window_none이 다수면 unsupported(모든 창을 확인한 뒤에만), no_tool/clarify가 다수면 그 결정, 동률이면 delegate다.
- delegate는 prediction-v1 결정이 아니다. 정확도에서는 오답이고, coverage(=1−delegate 비율)로 따로 보고한다. 어떤 결정의 TP/FP도 아니며 호출로 세지 않는다. confidence 임계값으로 clarify를 만들지 않고, 창 누락을 unsupported로 바꾸지 않는다. 후보 밖 선택은 구조상 0이어야 하므로, 1건이라도 나오면 오류로 중단한다.
- 공유 `jery.evaluation.metrics.evaluate`와 prediction-v1은 바꾸지 않았다. case 지표는 `laya_full.case_metrics`가 별도로 계산한다.

## 출력 인터페이스

`outputs/model/laya-full-20261017/<run>/`:

| 파일 | 내용 |
|---|---|
| `status.json` | running / succeeded / failed, pid, 오류 |
| `run.log` | 사람이 읽는 진행 로그 |
| `token_budget.json` | split별 요약과 잘림 창 목록(있으면 학습 안 함) |
| `before_validation.json` | 학습 전 validation(창·case) |
| `metrics.jsonl` | epoch 1..64마다 1줄. 매 줄 즉시 flush·fsync. `{epoch, train_loss, train_loss_last, steps, train_seconds, validation: {case_accuracy, decision_macro_f1, coverage, case: {...}, window: {...}, seconds}, time}` |
| `steps.jsonl` | optimizer step마다 `{step, epoch, loss, grad_norm, encoder_lr, head_lr, peak_memory_gb, seconds}` |
| `report.json` | `{config, provenance, export, data, parameters, token_budget, before_validation, train, updates, validation, test, restore, memory, wall_seconds, command}` (probe는 `probe.json`) |
| `validation_cases.jsonl`, `validation_windows.jsonl`, `test_cases.jsonl`, `test_windows.jsonl` | case별 결정·도구·경로·창 선택·라운드, 창별 확률 |

`case_accuracy`는 결정이 맞고, select_tools이면 도구도 허용 대안에 속할 때만 정답으로 센다. `case`에는 decision_accuracy, decision별 P/R/F1, tool_accuracy, unnecessary_call_rate, out_of_candidate_selections, delegated, coverage, 경로별 건수, 평균·최대 forward 수와 라운드 수가 들어간다. `window`에는 any-of top-1, soft CE(1e-12 하한), gold 종류별 정확도가 들어간다.

`models/laya-full-20261017/<run>/`:
- `trainer/checkpoint-<step>/`: epoch마다 head와 AdamW 상태. 최신 2개를 유지한다.
- `final/trainable.safetensors` + `head.json`: 64 epoch head(약 59 MB, fp32). base 643 MB는 복제하지 않는다.

복원 검사: 새 base에 `final`을 load하고 validation 전체를 다시 평가한다. 마지막 epoch의 창 확률 최대 차이가 1e-6 이하이고 case 예측이 같아야 하며, encoder 해시도 같아야 한다. 그다음 test를 1회 평가한다.

재개: `--resume`(같은 run-id, 같은 config)으로 최신 checkpoint에서 이어 간다.
- native checkpoint(`checkpoint-<step>/`)에는 head(`trainable.safetensors`), AdamW moment(`optimizer_state.safetensors`), `trainer_state.json`(global_step, epoch, log_history)이 들어 있다. dropout RNG는 native에 없다. native는 `seed + global_step`으로 다시 seed하므로("Dropout draws are not replayed") 재개 후 step이 미세하게 달라졌다(tiny에서 final head 최대 \|Δ\| 2.0e-4). 그래서 `on_save`가 RNG key를 `mx_random_key.json`에 저장하고, 재개 시 `mx.random.seed((hi << 32) | lo)`로 복원한다. 그 결과 재개한 run이 중단 없는 run과 비트 단위로 같다(아래 표).
- 로그 정책: 재개 시 `steps.jsonl`과 `metrics.jsonl`에서 checkpoint 이후의 줄은 지우고, 다시 실행한 step의 줄만 남긴다. 따라서 step과 epoch는 각각 한 줄이다. checkpoint는 그 epoch의 validation보다 먼저 저장되므로, validation 기록이 없는 epoch는 재개 직후 평가한다(`resumed_validation: true`). 재개 전 중단된 epoch의 `train_loss`는 재개 후 다시 실행한 step만 평균한다.
- 학습 후 단계: 최종 head를 저장한 뒤 복원·test 중 실패하면 `--resume`이 학습 없이 checkpoint를 load한다. 이어서 마지막 epoch validation, 저장, 복원 비교를 다시 한다. test 결과는 계산 즉시 `test.json`(head SHA-256 포함)에 저장한다. 재개 시 `test.json`이 있으면 다시 점수화하지 않고 재사용하며, head SHA가 다르면 거부한다. `report.json`이 이미 있는 run은 `--resume`을 거부해 기존 report를 보존한다.
- `os._exit`나 kill로 끝나면 `status.json`은 `running`으로 남는다. supervisor는 `pid`가 살아 있는지 함께 확인해야 한다.

## 실행 명령

```sh
# 단위 테스트 (stdlib)
PYTHONPATH=src python3 -m unittest tests.model.test_laya_full tests.model.test_laya_pilot tests.evaluation.test_metrics

# 8 step probe (1회 실행함)
PYTHONPATH=src HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false caffeinate -i \
  tmp/venv-decision/bin/python -m jery.model.laya_full --config configs/model/laya-full-20261017.json \
  --assets-root /Volumes/Dove-Nest-SSD/projects/Jery --run-id probe-8step-s3407 --probe

# 본 학습 child (Codex supervisor가 순차 실행; GPU 동시 학습 금지)
PYTHONPATH=src HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false caffeinate -i \
  tmp/venv-decision/bin/python -m jery.model.laya_full --config configs/model/laya-full-20261017.json \
  --assets-root /Volumes/Dove-Nest-SSD/projects/Jery --run-id lr1e-4-s3407 --override '{"head_learning_rate": 0.0001}'
# 중단 시 같은 인자 + --resume
```

작업 디렉터리에 `unsloth_compiled_cache/`가 생기므로 root를 cwd로 쓰지 않는다. 완료 여부는 `status.json`으로 확인한다(pgrep 자기 매칭 대기 금지).

## probe 결과 (실측)

환경은 파일럿과 같다(`tmp/venv-decision`, unsloth/unsloth-zoo 2026.10.6, mlx 0.32.3, torch 2.14.1, transformers 5.17.0). 모델은 root `models/laya-multilingual@1720e3e3`(읽기 전용)이고, 작업 브랜치는 dev-claude HEAD `c877fef`(미커밋 변경)이다. export SHA-256은 train `42a287bd…`, validation `aa2c7175…`, test `1e13e7a5…`이며 manifest와 일치한다.

1. `probe-8step-s3407` (실패, 기록 보존): 토큰 예산을 통과한 뒤 학습 전 validation 평가 중 자체 메모리 가드(20 GiB)에 걸려 중단됐다. 학습 step은 0회였다. 원인은 배치마다 패딩 길이가 달라 MLX 버퍼 캐시가 누적된 것이다. validation 320창에서 캐시 상한이 없으면 cache가 24.8 GB(active 0.67 GB)까지 늘었고, 상한 2 GiB에서는 2.16 GB였으며 시간은 같았다(25.7 s → 23.8 s). 그래서 CLI 시작 시 `mx.set_cache_limit(mlx_cache_limit_gib=2)`를 두었다. 프로세스 설정이며 커널 설정은 바꾸지 않았다.
2. `probe-8step-s3407-try2` (성공, 298.7 s):

| 항목 | 값 |
|---|---|
| 토큰 예산 (7,341창, 5.6 s) | 잘림 0. 시퀀스 최대 950 / 953 / 930 (train / val / test), 중앙값 687–689, 옵션 최대 15토큰, 지침 48토큰, state 최대 797, 여유 최소 868 → export 수정 불필요 |
| 학습 파라미터 / dtype | 14,770,945 / 321,710,593. encoder 89개 float16 + 45개 float32 텐서(고정), head 24 · scorer 6 · type_emb 1 float32 |
| optimizer step (microbatch 1 × 4) | 중앙값 0.509 s, 최대 0.558 s (첫 step 0.58 s) |
| 8 step | loss 3.123 → 2.302, head 31/31 텐서 갱신(최대 \|Δ\| 4.0e-4), encoder 해시 동일 |
| validation 전체 평가 | 140.8 s (창 1,592개 125.9 s + 결승 라운드), case당 평균 7.57 forward, 최대 15, 라운드 최대 2 |
| epoch 추정 | 학습 531 s + validation 141 s = 672 s |
| child 추정 (64 epoch) | **11.9 시간**, 3 child 순차 약 35.8 시간 + child당 복원·test 약 5분(test 평가 추정 139 s) |
| 메모리 | 학습 MLX peak 1.37 GB, 평가 peak 약 2.2 GB, 가드 peak(RSS·MLX 중 큰 값) 4.50 GiB |
| 학습 전 validation (기본 head) | case_accuracy 0.1613, decision macro-F1 0.2825, coverage 1.0, 경로 single 124 / final 123 / vote 1, 후보 밖 0, 창 any-of 0.0879 |

학습 전 값은 시작점일 뿐이다. 8 step 뒤의 case 지표는 같았다(0.1613). 이 수치는 품질 주장이 아니다.

## tiny end-to-end 검증 (경로 검증 전용)

fixture: `tmp/laya-tiny-20261017/export/`(무시 경로). 동결된 full export에서 단일 창(|C| ≤ 8) 사례를 split별로 결정 종류마다 하나씩 먼저 고르고 나머지를 원본 순서로 채웠다. train 8창(select 5, clarify / unsupported / no_tool 각 1), validation 3, test 3이다. manifest에는 새 SHA-256(train `d02321c6…`, validation `0cea7c8b…`, test `e99c507f…`)과 원본 manifest SHA(`ded2f8e2…`)를 기록했다. 다창 결승 경로는 이 fixture에 없다. 그 경로는 probe의 전체 validation(final 123 case)과 단위 테스트로 확인했다.

설정은 본 학습과 같고 `epochs=2`만 다르다(train 8 / accum 4 = 2 step/epoch, 4 optimizer step). base는 root `models/laya-multilingual`(읽기 전용), 환경은 `tmp/venv-decision`이다. 실행기 `tmp/laya-tiny-20261017/run.sh <run-id> [--resume]`는 override로 export와 출력 경로만 바꾼다. 강제 중단은 test 전용 `tmp/laya-tiny-20261017/crash_wrapper.py`가 `LAYA_CRASH_AT`(epoch1 / step3 / head / test) 시점에 `os._exit(99)`로 처리한다. production 코드에는 test mode를 넣지 않았다. 비교는 `tmp/laya-tiny-20261017/compare.py`가 checkpoint 텐서, optimizer, final head, step loss, 예측을 대조한다.

| run | 시나리오 | 결과 |
|---|---|---|
| `tiny-uninterrupted`, `tiny-resume`(같은 설정 반복) | 중단 없이 2 epoch | 각 14 s. checkpoint-2와 checkpoint-4 생성, epoch 1·2 validation 각 1줄, 최종 head 복원 차 0.0, test 1회. 두 run이 모든 텐서·loss·예측에서 0.0 차이로 결정적임 |
| `tiny-resume2` (RNG 저장 전 코드) | epoch 1 줄 flush 직후 중단 → `--resume` | 진짜 재개: step 2에서 시작, step 1–2 재실행 없음, epoch 1·2 각 1줄. checkpoint-2 텐서와 AdamW는 0.0 차이였으나, step 3·4 loss가 −0.0157 / −0.0773, final head 최대 \|Δ\| 2.0e-4, 예측 선택은 같음 → RNG 미복원이 원인 |
| (실패, 산출물 삭제) | 첫 RNG 복원 시도 | `mx.random.state[0] = …`가 mlx 0.32.3에서 TypeError → `seed((hi<<32)\|lo)` 방식으로 수정(동일 난수열을 직접 확인) |
| `tiny-rng-uninterrupted` vs `tiny-rng-resume` | RNG 저장·복원 후 epoch 1 중단 → 재개 | 모든 checkpoint 텐서, optimizer, final head, 4개 step loss, validation·test 확률 **0.0 차이** |
| `tiny-midepoch` | step 3 줄 flush 직후(epoch 2 중간) 중단 → 재개 | checkpoint-2에서 재개. 중단 전에 쓴 step 3 줄은 지우고 다시 실행한 줄만 남음 → steps.jsonl 4줄(step 1–4 각 1회). 결과는 중단 없는 run과 0.0 차이 |
| `tiny-post` | final/head.json 직후 중단 → 재개(test.json 직후 중단) → 재개 → 다시 재개 | 첫 재개: 학습 없이 checkpoint-4 load, 복원 비교와 test 저장 후 중단. 둘째 재개: "test already scored by this run; reusing test.json"로 test를 다시 점수화하지 않고 report 작성. test.json 바이트 동일, `report.test`와 일치. 셋째 재개: "already finished; its report is kept"로 거부(exit 2). 최종 결과는 중단 없는 run과 0.0 차이 |
| `tiny-schema` | epoch 1 중단 → 재개, report 키 확인 | 재개 run의 `report.json` 키가 중단 없는 run과 같음(`before_validation`을 파일에서 다시 읽도록 수정). 값 차이는 `seconds`뿐 |

출력 형식은 이 문서의 표와 같다. `metrics.jsonl` 한 줄은 `{epoch, train_loss, train_loss_last, steps, train_seconds, time, validation: {case_accuracy, decision_macro_f1, coverage, case, window, seconds}}`이고, 재개 직후 평가한 줄에는 `resumed_validation`이 붙는다. `report.json`에는 표의 키 외에 `run_id`, `probe`, `guard_peak_gib`가 있다. checkpoint 하나는 head 59 MB + AdamW 118 MB이며 최신 2개만 유지한다. 이 tiny run들의 가중치(무시 경로 `models/laya-tiny-20261017/`, 약 3.1 GB)는 지워도 된다.

## 검증 범위와 남은 위험

- 실행한 것: `tests.model.test_laya_full` 6개(집계 R0–R4, 라벨 차단, delegate 지표, 후보 밖, 창 지표)를 포함해 모델·평가 단위 테스트 29개 통과, 추론 전용 smoke(검증 6 case, 39창; 배치 점수와 `FastDecisionModel.predict`의 top-1 일치, 확률 최대 차 4.9e-5), probe 1회 성공.
- probe가 실행하지 않은 경로(epoch checkpoint, 학습 도중 validation, `--resume`, 최종 head 저장·복원, test 1회)는 아래 tiny fixture로 실제 실행해 검증했다. tiny 결과는 경로 검증일 뿐이다. 전체 학습을 대신하지 않으며, MLflow에 등록하지 않는다.
- 복원 허용 오차: 창 확률 차 1e-6 이하, case 예측 동일. 배치 평가와 단건 `predict` 사이에는 4.9e-5 차이가 있으므로, 비교는 같은 배치 경로끼리 한다.
- window_none이 train 창의 82%다. 창 정확도는 편향되므로 case 지표를 주 지표로 쓴다.
- 결승 창은 학습 분포(원래 창)와 구성이 다르다(승자만 모임). 이 영향은 측정하지 않았다.


## 변경 파일

- `src/jery/model/laya_full.py` (신규): CLI, label-blind 집계, case·window 지표, 학습·재개·복원·test.
- `src/jery/model/laya_pilot.py`: `load_export`가 split 목록을 받도록 일반화(기본값은 기존 동작).
- 이번 보강(`src/jery/model/laya_full.py`): 재개 시 checkpoint 이후 step·epoch 줄 정리, dropout RNG key 저장·복원(`on_save`, `mx_random_key.json`), `test.json` 즉시 저장과 재사용, 완료된 run의 `--resume` 거부, 재개 report의 `before_validation` 복원.
- tiny 검증 도구(무시 경로): `tmp/laya-tiny-20261017/{export/,run.sh,crash_wrapper.py,compare.py}`, 결과는 `outputs/model/laya-tiny-20261017/<run>/`.
- `tests/model/test_laya_full.py` (신규): 집계 규칙 R0–R4, 라운드, 라벨 차단, delegate 지표, 후보 밖 계산, 창 지표.
- probe 산출물: `outputs/model/laya-full-20261017/probe-8step-s3407/`(실패 기록), `outputs/model/laya-full-20261017/probe-8step-s3407-try2/`(`probe.json`, `before_validation.json`, `token_budget.json`, `steps.jsonl`, `run.log`, `status.json`). probe는 가중치를 저장하지 않는다.
- `configs/model/laya-full-20261017.json`, `configs/model/laya-full-20261017-children.json` (신규).
- 공유 `jery.evaluation.metrics`와 root 파일은 변경하지 않았다. 새 의존성은 없다(검사 중 실수로 venv에 설치한 pyflakes는 즉시 제거했다).

## 통합 검증

`dev-codex` 통합 시 보고서는 예측 파일 작성 후 atomic replace로 확정하도록 보완했다. 재시작 시 마지막 미완성 JSONL 줄만 복구하며, 완성된 줄의 JSON 오류는 그대로 거부한다. 해당 복구 테스트를 포함해 root 모델 27개·평가 3개 테스트가 통과했다. 데이터 22개, 검토 37개, 통합 15개를 합해 104개 테스트가 통과했다. `graphify update .`도 실행했다.
