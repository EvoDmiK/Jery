# Laya Multilingual head-only 공학 파일럿 (2026-10-17)

`decision-export-v1` 32/12행으로 기존 Unsloth MLX 결정 학습 경로를 M4에서 실제로 실행했다. 목적은 **경로 검증**이다. 미검토 합성 데이터 32행이므로 정확도·보정·운영 채택 근거로 쓰지 않는다. MCP 도구는 실행하지 않았고, 공개 prediction-v1과 root 파일·데이터는 바꾸지 않았다.

결론: 환경, 토큰 예산, 고정 본체, 유한 손실·기울기, head 갱신, 저장·재로딩 일치, 지연·메모리 측정이 모두 실제로 통과했다. 원래 설정(head lr 1e-4, 64 step)은 손실만 조금 낮추고 학습 top-1은 그대로였다. 즉 "train 정답 일치 상승" 양성 대조는 **원래 run에서 충족되지 않았다**.

## 환경과 출처

| 항목 | 값 |
|---|---|
| 하드웨어 | Mac16,10 (M4), 32 GiB, macOS 26.6.2 arm64 |
| venv | `tmp/venv-decision` (Python 3.12.15, 무시 경로). root `.venv`는 건드리지 않음 |
| 패키지 | unsloth 2026.10.6, unsloth-zoo 2026.10.6, mlx 0.32.3, mlx-lm 0.31.3, torch 2.14.1, transformers 5.17.0, tokenizers 0.23.3, laya 0.4.9 ([requirements-decision.txt](../../requirements-decision.txt)) |
| 소스 대조 | 설치된 `decision_mlx.py`, `_decision_common.py`는 unsloth@`2f63c766`과 바이트 동일. zoo `mlx/decision.py`는 CRLF 정규화 후 zoo@`9e560677`과 동일 |
| compile 패치 | zoo `mlx/trainer.py`의 PyPI와 main 차이는 `_local_grad_step`(LM trainer)에서 `mx.stop_gradient`를 `+ 0`으로 바꾼 mlx 0.32.4 우회뿐이다. `MLXDecisionTrainer`는 이 함수를 쓰지 않고 mlx는 0.32.3이므로 패치하지 않았다 |
| mlx 버전 | 검토 문서의 0.32.4가 아니라 unsloth-zoo 2026.10.6의 darwin-arm64 고정값 0.32.3이다 |
| 모델 | root `models/laya-multilingual` revision `1720e3e3357cfe1e281542e223f8273b0890ca34`, 읽기 전용, 복사 없음. `model.safetensors` SHA-256과 HF 다운로드 메타데이터를 report에 기록 |
| export | root `outputs/laya-pilot-20261017/export`, manifest SHA-256 검증 후 사용. train sha `10769b08…`, validation sha `a2bfd802…` |
| 라이선스 | 결정 학습 코드는 AGPL-3.0-only. 로컬 실험으로만 사용 |

G0: `FastDecisionModel.from_pretrained`가 체크포인트를 1.7초에 로드했고, 예제 1건 predict가 성공했다(MLX peak 0.71 GB).

## 실행 방법

```sh
# 환경 (한 번)
python3.12 -m venv tmp/venv-decision  # CPython 3.12.15
tmp/venv-decision/bin/pip install -r requirements-decision.txt

# 파일럿 (재현용, 현재 CLI)
PYTHONPATH=src HF_HUB_OFFLINE=1 USE_TF=0 TOKENIZERS_PARALLELISM=false caffeinate -i \
  tmp/venv-decision/bin/python -m jery.model.laya_pilot \
  --config configs/model/laya-pilot-20261017.json \
  --assets-root <integration-root> --run-id <new-id>
```

- 설정 [configs/model/laya-pilot-20261017.json](../../configs/model/laya-pilot-20261017.json)은 export와 모델 경로를 `--assets-root` 기준 상대 경로로 둔다. `--override '{"key": value}'`는 설정 위에 병합되며 report의 `config`에 남는다.
- 같은 run id가 있으면 실행을 거부한다. 출력은 `outputs/model/laya-pilot-20261017/<run>/`에 쓴다: `status.json`(running/succeeded/failed), `run.log`, `report.json`, `steps.json`, `budgets.json`, `predictions_{before,after,reloaded}.jsonl`. 가중치는 `models/laya-pilot-20261017/<run>/`에 쓴다.
- 실제 두 run은 위 CLI의 이전 상태로 실행했다. run 1은 `--assets-root`/`--override`가 없고 설정에 절대 경로가 있었다. run 2에는 `--override`가 있었다. 각 report의 `command`와 `config`가 실제 값이다. `--assets-root`는 두 run이 끝난 뒤 추가했으며, 재학습으로 검증하지 않았다.
- run 1은 `nohup caffeinate -i …`로, run 2는 `caffeinate -i …`로 직접 실행했다. coordinator가 권한 root `scripts/run_local_training.py`는 사용하지 않았다. 그 대신 CLI 자체의 status.json, run.log, 메모리 가드로 같은 정보를 남긴다.
- unsloth를 import하면 작업 디렉터리에 `unsloth_compiled_cache/`가 생긴다. 생성물이므로 삭제했다.
- 실행 당시 브랜치는 `dev-claude` HEAD `c877fef`이고, 파일럿 파일은 아직 커밋하지 않았다(report의 `git.dirty=true`).

## 고정 설정 (run 1, 원래 설정)

seed 3407, `max_seq_length` 1024, head 예산 256, microbatch 1 × accum 4, 64 optimizer step(8 step/epoch × 8 epoch), AdamW(bias correction), weight decay 0, 전역 grad-norm clip 1.0, linear schedule, warmup 4, gradient checkpointing on, MLX compile 기본값 on, MLX 프로세스 메모리 상한 16 GB(`memory_limit_gb`; wired limit은 바꾸지 않음, 커널 설정 변경 없음), 자체 메모리 가드 20 GiB(RSS와 MLX active+cache 중 큰 값, 초과 시 status=failed 후 종료).

## 측정 결과

### 입력·파라미터·dtype

- 실제 Laya tokenizer로 `laya.common.build_sequence`를 재현했다. 재현한 ids가 원본과 같은지 assert로 확인했다. 결과적으로 옵션·지침·state 잘림 0건, 옵션 텍스트 충돌 0건이다. 시퀀스 길이는 train 최대 898 / 중앙값 606, validation 최대 767 / 중앙값 623이다. 옵션 최대 15토큰(마커 포함), state 여유 최소 868토큰이다. 그래서 export 수정 요청은 없다.
- 학습 가능한 파라미터는 14,770,945 / 321,710,593이다(`head`, `scorer`, `type_emb`). encoder 파라미터의 기울기 텐서는 0개다.
- dtype(로드 직후와 복원 후 동일): encoder 텐서 89개 float16 + 45개 float32, head 24 · scorer 6 · type_emb 1 텐서는 float32. 저장된 `trainable.safetensors`는 31개 텐서 모두 float32다.
- 실제 LR: 로그의 `learning_rate`는 encoder 그룹 값(2e-5 × scale)이다. 학습할 encoder 파라미터는 없다. head LR은 `head_lr × scale`이고, scale은 step 1에서 0, step 5에서 1.0(warmup 4)이며, 이후 linear로 내려가 step 64에서 1/60이다. 따라서 run 1의 head LR은 0 → 1e-4 → 1.7e-6이다.

### 두 run

| 항목 | run 1 `head-only-s3407-try1` (원래, head lr 1e-4) | run 2 `head-only-s3407-lr1e-3-try2` (head lr 1e-3) |
|---|---|---|
| 첫 microbatch 기울기 | loss 5.447, 전역 norm 48.46, 유한 | 동일 |
| step loss 처음 → 마지막 | 2.423 → 2.321 | 2.423 → 1.955 |
| 처음 8 / 마지막 8 step 평균 | 2.373 / 2.034 | 2.308 / 1.615 |
| grad norm 처음 → 마지막 | 9.80 → 5.70 | 9.80 → 3.35 |
| head 텐서 변경 / 최대 \|Δ\| | 31/31, 0.00286 | 31/31, 0.0275 |
| encoder SHA-256 전후 | 동일 (`a39f3030…`) | 동일 |
| train top-1 (any-of) 전 → 후 | 0.3125 → **0.3125** | 0.3125 → 0.500 |
| train soft CE 전 → 후 | 2.387 → 2.032 | 2.387 → 1.492 |
| train 정답 확률 질량 평균 전 → 후 | 0.233 → 0.207 | 0.233 → 0.298 |
| train 평균 confidence 전 → 후 | 0.592 → 0.431 | 0.592 → 0.396 |
| 바뀐 top-1 (train / validation) | 0 / 0 | 9 / 3 |
| 최대 \|Δ확률\| (train / validation) | 0.268 / 0.269 | 0.654 / 0.482 |
| validation top-1 전 → 후 (12행) | 0.000 → 0.000 | 0.000 → 0.000 |
| validation soft CE 전 → 후 | 4.133 → 3.122 | 4.133 → 2.941 |
| 학습 시간 (64 step) | 33.0 s | 27.9 s |
| MLX peak (학습 / 전체) | 1.51 GB / 2.25 GB | 1.51 GB / 2.25 GB |
| max RSS / 가드 peak | 5.51 GB / 5.13 GiB | 5.30 GB / 4.80 GiB |
| cold load / 복원 load | 1.70 s / 2.48 s | 1.61 s / 2.38 s |
| 저장 → 재로딩 | top-1 동일, 최대 \|Δ확률\| 0.0, encoder 동일 | 동일 |
| warm 지연 (validation 12행 × 5, 1 결정/호출) | p50 82.7 ms, p95 106.4 ms, max 111.8 ms | p50 79.2, p95 102.7, max 121.1 ms |
| head 산출물 | `trainable.safetensors` 59,086,741 B, sha `563830c6…` | 같은 크기, sha `90f82f85…` |
| 전체 wall | 68.5 s | 58.3 s |

run 1의 train 정답 유형별 결과: tool 7/12, clarify 2/5, no_tool 0/6, window_none 1/9이며 학습 전후 같다. run 2에서는 no_tool 4/6, window_none 3/9로 올랐고 tool·clarify는 같다. 두 run 모두 validation 12행에서 맞힌 행이 없다.

해석(사실 범위만):

- 원래 설정의 64 step은 head를 아주 조금만 움직였다. 손실은 줄었지만 어떤 행의 top-1도 바꾸지 못했고, confidence와 정답 질량이 함께 낮아졌다. 이 데이터와 step 수에서 head lr 1e-4는 양성 대조를 만들기에 부족했다.
- run 2는 coordinator의 05:49 "추가 튜닝 금지·재현성 진단만" 지시를 읽기 전인 05:51에 시작했다. 지시에 따라 결과를 보존해 보고만 하며, 이 run으로 설정을 선택하지 않는다. run 2는 학습셋 기억(train top-1 상승)이 가능하다는 것만 보여 준다. validation 0/12와 행 12개로는 일반화나 lr 우열을 말할 수 없다.
- 두 run의 MLflow 업로드는 coordinator가 담당한다(run 1 완료 통보 받음). 이 워커는 MLflow를 변경하지 않았다.

## 산출물

- 코드: [src/jery/model/laya_pilot.py](../../src/jery/model/laya_pilot.py), 테스트 [tests/model/test_laya_pilot.py](../../tests/model/test_laya_pilot.py)(any-of 정확도와 soft CE, export 해시 불일치 거부, stdlib만 사용).
- run 1: `outputs/model/laya-pilot-20261017/head-only-s3407-try1/`, `models/laya-pilot-20261017/head-only-s3407-try1/{trainable.safetensors,pilot.json}`.
- run 2: `outputs/model/laya-pilot-20261017/head-only-s3407-lr1e-3-try2/`, `models/laya-pilot-20261017/head-only-s3407-lr1e-3-try2/`.
- 복원 방법: `FastDecisionModel.from_pretrained(<base>)`를 호출한 뒤 `unsloth_zoo.mlx.decision.load_trainable(model, <run dir>)`를 적용한다. head만 저장하므로 643 MB 본체는 복제하지 않는다. merged 16-bit 저장(`save_pretrained_merged`)은 본체 전체를 다시 쓰므로 사용하지 않았다.

## 남은 일 (실행하지 않음)

- 양성 대조 미충족 처리: 같은 32행에서 step 수나 head lr을 사전 등록하고 한 번 비교할지, 데이터를 키운 뒤 판단할지 coordinator가 결정한다. 이번 작업에서는 추가 학습을 하지 않는다.
- option 순서 셔플: Laya MLX 경로의 `permute_fields`는 Clef 전용이다. export의 옵션 순서는 고정(T0…, sentinel 끝)이므로 위치 편향 검증이 필요하다.
- root `scripts/run_local_training.py` 실행기로 감싸서 재현할 것.
- window→공개 prediction-v1 변환(C 전체 확인 후 unsupported, 위임)과 OOF 후보, LoRA 단계, 보정은 범위 밖이다.
