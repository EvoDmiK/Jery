# 결정 모델 학습 전략 독립 검토 (dev-claude, 2026-10-17)

설계 검토 문서다. 학습·설치·대용량 다운로드·제품 코드 변경은 하지 않았다. 아래에서 **[출처]**는 고정 revision의 원문·소스에서 확인한 사실, **[측정]**은 이 저장소에서 직접 계산한 값, **[추정]**은 검증되지 않은 엔지니어링 추정이다. 추정치는 파일럿 게이트로만 사용한다.

## 0. 결론 요약

1. **첫 전용 결정 모델 후보는 Laya Multilingual FP16 LoRA가 맞다.** 다만 근거는 "Laya가 더 정확하다"가 아니다. M4에서 가장 싸고 빠르게 실패 여부를 판정할 수 있는 경로이기 때문이다. 이번 파일럿 결과로 프로덕션 모델을 정하지 않는다.
2. **Laya의 옵션 예산은 실제 소스로 확인한 하드 제약이다.** 옵션 하나에 최대 48토큰, 질문과 옵션 전체에 256토큰이 배정된다. 옵션이 12개(도구 8 + 공개 sentinel 3 + 내부 `none_of_these`)면 옵션당 20토큰으로 잘린다. 따라서 도구 명세는 옵션이 아니라 state에 압축해서 넣는다. 옵션에는 짧은 alias와 한 줄 요약만 둔다. 실제 Laya tokenizer로 센 토큰 수를 G1 게이트로 둔다.
3. **Gemma 4 E4B 4bit + Clef 헤드는 비교군으로 유지하되 Laya 게이트 뒤로 미룬다.** 경로 자체는 소스에 존재한다(`load_language_model_as_clef(..., load_in_4bit=True)`). 그러나 다음 세 가지가 미확인이다. (a) Gemma 4로 실제 테스트된 적이 없다. zoo의 Metal 테스트는 작은 qwen3_5 decoder만 사용한다. (b) Clef 프롬프트에 Qwen 대화 토큰(`<|im_start|>`, `<think>`)이 하드코딩되어 있다. (c) 이미 받아 둔 mlx-community 4bit 폴더를 이 경로가 읽을 수 있는지 모른다. 읽지 못하면 BF16 원본(약 16GB) 다운로드가 필요하다. 6건 생성 smoke 결과는 결정 헤드 성능과 무관하므로 Gemma 탈락 근거로 쓰지 않는다.
4. **BF16 Gemma는 이번 범위에서 제외한다.** `Clef on MLX trains at its checkpoint's precision`[출처]이므로 BF16 학습은 BF16 체크포인트를 그대로 올리는 것이다. 32GiB 통합 메모리에서 가중치 약 16GB에 활성값과 시스템 여유분까지 더하면 실패 위험이 높다[추정]. 4bit 경로가 메모리 때문에 실패했다는 측정 근거가 생기면 그때 재검토한다.
5. **Shortlist 누락은 절대 `unsupported`로 바뀌지 않는다.** 원 후보 집합 C와 내부 shortlist S를 분리한다. 헤드에는 공개 결정과 별개인 내부 전용 옵션 `none_of_these`를 하나 더 둔다. 이 옵션이 나오면 다음 창으로 넘어가 C 전체를 확인한다. 공개 `unsupported`는 C 기준 정답에서만 학습하고, C 전체를 확인한 뒤에만 낸다(§4).
6. **대안 정답 손실**: MLX 트레이너는 soft cross-entropy(`-(target·log p)`)를 쓴다[출처]. 허용 대안이 2개인 행에 0.5/0.5 target을 주면 "둘 다 정답"이 아니라 "확률을 반반 나눠라"를 학습한다. 해당 행은 train 12행(2%)뿐이므로 균등 target을 쓰고 평가는 any-of로 한다. any-of 손실(`-log Σ p_gold`)은 커스텀 코드가 필요하므로 도입하지 않는다.
7. **보정은 Jery가 직접 소유한다.** Unsloth Clef 보정은 hard label 기준이라 대안 정답 행의 다른 정답을 오답으로 센다[출처]. Laya 보정은 질문 유형별 온도만 맞춘다. 결정 유형별 표본이 매우 적다(검증 비선택 16건). 그래서 보정용 분할을 따로 두고 any-of 정답 기준으로 온도 하나만 맞춘다. 결정 유형별 임계값은 근거가 생기기 전까지 쓰지 않는다.

## 1. 확인한 출처와 revision (2026-10-17 기준)

| 대상 | 고정 revision | 확인 내용 |
|---|---|---|
| [unsloth](https://github.com/unslothai/unsloth/tree/2f63c766cca299de97f71434796cd853d49036b9) | `2f63c766` (2026-10-17) | [`unsloth/models/decision_mlx.py`](https://github.com/unslothai/unsloth/blob/2f63c766cca299de97f71434796cd853d49036b9/unsloth/models/decision_mlx.py), [`_decision_common.py`](https://github.com/unslothai/unsloth/blob/2f63c766cca299de97f71434796cd853d49036b9/unsloth/models/_decision_common.py), Studio 기본값 [`convaiinnovations_laya.yaml`](https://github.com/unslothai/unsloth/blob/2f63c766cca299de97f71434796cd853d49036b9/studio/backend/assets/configs/model_defaults/decision/convaiinnovations_laya.yaml)·[`llm_decision_defaults.yaml`](https://github.com/unslothai/unsloth/blob/2f63c766cca299de97f71434796cd853d49036b9/studio/backend/assets/configs/model_defaults/decision/llm_decision_defaults.yaml), vendored [`laya/common.py`](https://github.com/unslothai/unsloth/blob/2f63c766cca299de97f71434796cd853d49036b9/studio/backend/vendor/laya/common.py) |
| [unsloth-zoo](https://github.com/unslothai/unsloth-zoo/tree/9e560677fbe7460cf9d2e047d8825689cacb02b7) | `9e560677` (2026-10-16) | [`unsloth_zoo/mlx/decision.py`](https://github.com/unslothai/unsloth-zoo/blob/9e560677fbe7460cf9d2e047d8825689cacb02b7/unsloth_zoo/mlx/decision.py), `mlx/trainer.py`, `tests/test_mlx_decision_metal.py` |
| PyPI | unsloth `2026.10.6`, unsloth-zoo `2026.10.6`, mlx `0.32.4` | 최신 릴리스가 위 main 커밋을 포함하는지는 **미확인**이다. 설치 후 소스 대조를 G0 게이트로 둔다. |
| [laya-multilingual](https://huggingface.co/convaiinnovations/laya-multilingual) | `1720e3e3` (2026-10-03) | mmBERT-base 307M + 결정 헤드, 총 322M. 기본 1024토큰, 질문·옵션 256토큰. "~20개 옵션 이하" 권장. 출시 상태는 미보정(T=1.0). MASSIVE 한국어 intent 0.490. Apache-2.0 |
| [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya/tree/4ac226d5ee88ed6df8db288a05c766d7a34ee682) | `4ac226d5` (2026-10-15) | [`docs/finetune.md`](https://github.com/NandhaKishorM/laya/blob/4ac226d5ee88ed6df8db288a05c766d7a34ee682/docs/finetune.md): `laya-train`, `--device auto`(CUDA/MPS/CPU), 기본 4 epoch, micro 8 × accum 8, encoder 2.5e-5 / head 1e-4, `rlcd` 또는 `soft-ce` 손실, `--freeze-encoder`, `--shuffle-options`. Apple 학습 메모리·시간 수치는 없다. 저장소의 MPS 수치는 추론용 fp16 autocast 벤치마크(M5 Pro)다. |
| [unsloth/gemma-4-E4B-it](https://huggingface.co/unsloth/gemma-4-E4B-it) | `4e22d7e5` | `gemma4_text`, hidden 2560, 42층, vocab 262,144. Safetensors 전체 약 8.0B 파라미터(비전·오디오 포함) |
| 문서 | [decision model guide](https://unsloth.ai/docs/basics/train-your-own-decision-model-with-unsloth.md), [decision-laya](https://unsloth.ai/docs/models/decision-laya.md) | 가이드에는 Apple Silicon/MLX 언급이 없다. 서빙 문서는 "설명 붙은 옵션이 ~20개를 넘으면 레이블이 잘린다"고 한다. 가이드 안에서도 lr이 2e-4(본문)와 2e-5(코드)로 서로 다르다. |
| 원문 정리 | `tmp/jev-decision-source.md` ([PyTorchKR](https://discuss.pytorch.kr/t/unsloth-llm-laya-clef-jev/12158)) | Nvidia VRAM·시간 표(Gemma 4 E4B 14.4GB/49분, Laya 2.5GB/10분)는 L4 등 GPU 수치이므로 M4로 옮기지 않는다. 결정 학습 소스의 라이선스는 `AGPL-3.0-only`다. |

### 소스에서 확인한 사실 (MLX 경로)

- **Laya**: 16-bit로만 학습한다. `load_in_4bit`는 "Laya decision models train in 16-bit, so load_in_4bit is not supported"로 거부된다. full finetune이 아니면 dtype은 float16으로 고정된다. 저장은 `merged_16bit`만 된다.
- **Clef(디코더)**: 4bit 로드 시 LoRA로만 학습한다. dtype 지정은 거부된다("trains at its checkpoint's precision").
- **LoRA 기본값**: `r=64`, `lora_alpha=64`, `target_modules="all-linear"`. `layers_to_transform`, `layers_pattern`, `use_dora`, `modules_to_save`, `bias`, `init_lora_weights`, `loftq_config`는 기본값이 아니면 **거부**된다. 따라서 상위 층만 LoRA를 거는 식의 설정은 불가능하다.
- **헤드**: 새 Clef 헤드의 width는 hidden ≥ 3072이면 1024, 아니면 512다. Gemma 4 E4B는 hidden 2560이므로 **512**다. 구성은 routing 2층, 4층, heads = width/64, FFN 4×width이고 float32로 학습한다. Laya 헤드는 사전학습된 헤드다. 따라서 Laya의 head-only 단계는 이미 학습된 헤드의 적응이지, 새 헤드 warmup이 아니다.
- **데이터**: `gold[name]`에는 label 또는 `{"label", "probabilities"}`를 넣는다. probabilities는 정규화되어 soft target이 된다. 질문별로 gold가 없으면 그 질문만 건너뛴다.
- **손실**: `-(target · log_softmax)`이며 선택적으로 label smoothing, Brier, ordinal, KL을 더한다. 대안 정답의 any-of 손실은 **없다**.
- **보정**: 질문 유형별 온도를 LBFGS로 맞추며 `[0.1, 10]`(Laya) 범위로 clamp한다. 해당 유형 표본이 `MIN_CALIBRATION_ITEMS = 10`보다 적으면 1.0을 유지한다. holdout은 10%, 최대 400행이다. Clef 보정은 "Calibrated against being right (the gold label), not the soft gold distribution"이다.
- **Laya 시퀀스**: `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`. 옵션 텍스트는 48토큰에서 자른다. 옵션 합이 `head_max_len`(학습 256) − 16을 넘으면 `per = (256-16)//n_options`로 균등하게 자른다. state는 나머지를 쓰며 끝부분부터 잘린다.
- **Clef 시퀀스**: 옵션 설명은 잘리지 않고 schema 전체가 `max_length` 안에 들어가야 한다(넘으면 오류). state만 잘린다. 시스템 프롬프트와 템플릿은 Qwen 형식 문자열이다.
- **Trainer**: `optim_args`, `auto_find_batch_size`, `save_strategy="best"`는 거부되고 MLX에 대응이 없는 인자는 무시된다(경고만 출력). evaluate와 calibrate의 `batch_size`는 사용되지 않는다(Clef는 한 행씩 처리).

## 2. 이 저장소 데이터로 측정한 사실

데이터 검증(`python3 scripts/validate_mcp_dataset.py`: PASS, 1162/96/5/340)을 먼저 실행했다. 아래 값은 train과 validation만으로 계산했다. test는 사용하지 않았다.

| 항목 | 값 |
|---|---|
| 도구 설명 글자 수 중앙값 / p90 / 최대 | 210 / 653 / 1,448 |
| input_schema JSON 글자 수 중앙값 / p90 / 최대 | 465 / 1,470 / 5,139 |
| 서버별 설명 중앙값(글자) | grafana 259, korean_law 258, obsidian 164, kestrel 139, swallow 93 |
| 요청+맥락 글자 수 중앙값 / 최대(train) | 37 / 148 |
| validation 후보 수 분포 | 96개 122행, 8개 96행, 7개 19행, 4개 6행, 5·6·9개 5행 |
| 대안 정답 2개 행 | train 12, validation 4 (모두 `swallow_check_stock`) |
| 선형 기준의 validation recall@k (선택 정답 232행) | @1 .750, @3 .927, @5 .970, **@8 .974**, @12 .983, @16 .987 |
| 같은 지표, 후보 96개 행(116행)만 | @1 .612, @5 .940, **@8 .948**, @12 .966, @16 .974 |
| 후보 96개 행에서 @8 누락 | 6건, 모두 Grafana |

해석:

- Shortlist 8을 적용하면 후보 96개 행에서 정답 도구의 약 5%가 헤드 입력에서 빠진다. 누락 6건은 모두 Grafana다. 그대로 두면 이 행들의 도구 정확도 상한은 94.8%다. 이 값은 validation이 이미 선형 설정 선택에 쓰였으므로 약간 낙관적이다.
- 후보가 9개 이하인 행은 shortlist가 원 후보 전체와 같아서 누락 문제가 없다. 누락 문제는 후보 96개 행에만 생긴다.
- 현재 선형 모델은 train에 맞춰 학습되었다. 따라서 train 행에 대한 shortlist는 거의 항상 정답을 포함한다. 헤드 학습용 shortlist는 반드시 scenario group 단위 out-of-fold로 만들어야 한다(§4).

## 3. Laya 옵션 예산과 입력 설계

Laya 학습 시 `head_max_len = 256`이고 옵션은 12개다(k=8 기준). 각 옵션은 `[MASK]` 1토큰과 텍스트로 이루어진다. 옵션 합이 240을 넘으면 옵션당 `(256-16)//12 = 20`토큰(텍스트 약 19토큰)으로 잘린다. k=12면 옵션 16개, 옵션당 15토큰이다. `mcp__grafana__list_prometheus_metric_metadata` 같은 전체 ID만으로도 10토큰을 넘을 수 있다[추정, mmBERT tokenizer로 측정 필요]. 그러면 설명이 들어갈 자리가 거의 없다.

권장 표현(Laya):

- **옵션**(k+4개): `"<alias>": "<서버>.<짧은 이름>: <설명 첫 구절>"`. alias는 `T1`…`T8`이나 짧은 도구명을 쓴다. 공개 sentinel은 `no_tool: 도구 없이 답변 가능`, `clarify: 필수 대상·정보가 빠져 확인 필요`, `unsupported: 제시된 후보 전체로 처리 불가`로 둔다. 내부 옵션은 `none_of_these: 이 목록 밖의 도구가 필요`로 둔다. alias와 원래 ID의 대응은 adapter가 보관하고, 출력은 원래 ID로 되돌린 뒤 prediction-v1로 검증한다.
- **state**(약 760토큰): `{"request", "context"(최근 N턴), "candidates": [{"alias", "server", "name", "summary", "action_enum"}]}`. summary는 설명 앞부분을 기준으로 잘라 후보 8개가 예산에 맞게 한다. 같은 alias를 옵션과 state 양쪽에 두어 헤드가 서로 대응시킬 수 있게 한다. state 끝부분부터 잘리므로 request와 context를 앞에 둔다.
- **옵션 순서**: 학습 시 도구 옵션 순서를 섞는다(Laya `--shuffle-options`, Unsloth `permute_fields`). sentinel은 끝에 고정한다.
- **action**: 같은 행에 두 번째 `choice` 질문을 둘 수 있다. 다만 도구가 정해져야 enum이 정해지므로 조건부 두 번째 호출로 처리한다. 선택된 도구 schema의 action enum을 옵션으로 렌더링한다. 옵션이 텍스트이므로 train에 없던 action도 점수화할 수 있다. 이 점이 닫힌 집합인 선형 action head(test 0.367)보다 나은 구조적 이점이다. seen과 unseen action 지표는 분리해서 보고한다.

Gemma/Clef는 옵션 설명이 잘리지 않으므로 shortlist 8개의 압축 명세를 옵션 description에 그대로 넣을 수 있다. 하지만 전체 schema가 `max_length` 안에 들어가야 한다. Gemma tokenizer 토큰 수를 Laya 토큰 수로 바꿔 쓰지 않는다. 두 tokenizer 모두 vocab 약 256k로 비슷해 보이지만 확인되지 않았다.

## 4. Shortlist와 누락 복구 (공개 계약을 바꾸지 않는 방식)

용어: `C` = 원 후보 집합(공개 입력), `S` = 선형 순위 상위 8개(내부 shortlist). `|C| ≤ 8`이면 `S = C`다.

추론:

헤드 옵션은 창의 도구 k개, 공개 sentinel 3개(`no_tool`, `clarify`, `unsupported`), 그리고 내부 전용 `none_of_these` 1개다. `none_of_these`는 "이 창의 도구로는 안 되지만, 창 밖 후보는 아직 보지 않았다"는 뜻이다. 공개 출력에는 절대 나오지 않는다.

1. 선형 라우터로 `C` 전체의 순위를 만든다. 확률은 곱하지 않고 순서만 사용한다.
2. 헤드는 창 `W1`(순위 1–k)을 평가한다. k는 G1 예산 측정에 따라 {8, 12} 중에서 정한다.
3. 헤드 출력에 따라 처리한다.
   - `select_tools`(창 안의 도구): 채택한다. alias를 원래 ID로 되돌린 뒤 validate한다.
   - `no_tool`, `clarify`: 요청 자체의 성질이므로 채택한다. 낮은 confidence를 `clarify`로 바꾸는 규칙은 두지 않는다.
   - `none_of_these`이고 아직 보지 않은 후보가 남음: 다음 창을 평가한다. `C`를 다 볼 때까지 반복한다. 후보 96개는 k=8이면 최대 12창이다[비용은 G3에서 측정].
   - 창 하나에서 `unsupported`가 나왔지만 C를 아직 다 보지 않은 경우: `none_of_these`와 똑같이 다음 창으로 넘어간다.
   - C의 모든 창이 `none_of_these` 또는 `unsupported`일 때만 공개 `unsupported`를 낸다.
   - 여러 창에서 도구가 선택되면, 창별 승자를 모아(≤ k) 한 번 더 평가한다. 서로 다른 창의 확률을 직접 비교하지 않는다.
4. 헤드 결과가 계약 검증에 실패하면 선형의 C 전체 결과로 넘긴다(delegate). 이것은 내부 fallback이다. 공개 decision 값을 새로 만들지 않으며, `clarify`나 `unsupported`로 대체하지도 않는다.

남는 위험: 창에 정답이 없는데 헤드가 `none_of_these` 대신 엉뚱한 도구를 고를 수 있다. 이 위험은 창 확장으로 막을 수 없다. recall@k와 `none_of_these` 학습으로 줄이고, 오류율을 따로 보고한다.

학습:

- shortlist는 scenario group 단위 5-fold out-of-fold 선형으로 만든다. fold마다 선형을 다시 학습한다. stdlib 기준 약 4분 × 5다[측정 기반 추정].
- **정답이 원 후보 C에 있는데 OOF shortlist에서 빠진 경우**: 학습 시에는 정답을 shortlist의 무작위 위치에 넣는다(가장 낮은 순위를 교체). 이렇게 하면 헤드가 "shortlist 누락 = unsupported"를 학습하지 않는다. 모델은 `unsupported`를 정답이 원 후보에 없는 경우에서만 배운다.
- **음성 창 학습**: 같은 요청에서 정답과 허용 대안을 모두 포함하지 않는 창(예: 순위 9–16)을 train에서만 만든다. 이 창의 target은 내부 `none_of_these`이며, 공개 라벨(`unsupported`)이 아니다. 공개 라벨은 계속 C에만 의존한다. 생성 규칙, 개수, 비율(양성 창 대비 1:1 이하)을 보고한다. 음성 창 없이 학습한 설정과 ablation으로 비교한다.
- **평가 게이트**: validation에서 정답이 C에 있는 행이 최종 `unsupported`가 된 건수를 창 확장 전과 후로 나눠 보고한다. 창 확장 후 그중 shortlist 누락이 원인인 건수는 **0**이어야 한다. "창에 정답이 없어 오선택"한 건수도 따로 보고한다. oracle shortlist(정답 강제 포함) 결과는 진단용으로만 쓰며 배포 성능으로 보고하지 않는다.

## 5. 손실, 대안 정답, 보정

- **대안 정답**: 대안이 있는 행의 gold는 `{"label": <첫 대안>, "probabilities": {alt: 1/k}}`로 둔다. 이것은 동등성의 인코딩일 뿐 사람이 측정한 확률이 아니다. soft CE 아래에서 최적 확률은 0.5/0.5이므로 이런 행은 confidence가 낮게 나온다. 평가는 any-of(top-1이 대안 중 하나)로 한다. 비율이 2%이므로 커스텀 손실의 이득보다 유지 비용이 크다. v3 데이터에서 대안이 많아지면 재검토한다.
- **Label smoothing과 Brier**: 기본값은 0으로 둔다. 각각 1개 값만 ablation한다.
- **보정**: 어떤 라이브러리의 보정 결과도 그대로 쓰지 않는다. Jery adapter가 logits를 받아 온도 T 하나를 맞춘다. 정답 기준은 any-of이며 shortlist 조건부다. 보정에 쓰는 데이터는 설정 선택에 쓰지 않은 분할이다(§7). 보정 표본이 적은 결정 유형(비선택)은 per-decision 온도나 임계값을 두지 않는다. 확률은 최종 창 기준의 조건부 확률로만 보고하며, 선형 점수와 곱하지 않는다.
- **보고**: 전체 ECE(15 bins), 결정 유형별 reliability(표본 수 포함), confidence ≥ 0.9인 행의 정확도와 coverage.

## 6. 비용과 자원 [추정, 파일럿에서 측정]

| 경로 | 다운로드 | 학습 메모리 | 1 epoch 시간(668행) | 근거 |
|---|---|---|---|---|
| Laya Multilingual FP16 LoRA (MLX 또는 laya-train MPS) | 약 0.7GB | < 8GB 예상 | 수 분~20분 | 322M, 시퀀스 1024. M5 Pro 추론 fp16에서 8행·2,258토큰 65ms[출처]. M4 기본 GPU 학습은 이보다 수 배 느릴 것 |
| Gemma 4 E4B 4bit + Clef LoRA (MLX) | 0 또는 약 16GB(BF16 원본 필요 시) | 10–20GB 예상 | 30–90분 | MLX 추론 피크 5.03GB(1.5–4.3k토큰)[출처: gemma-e4b-smoke]. 그래디언트 체크포인팅과 LoRA r16, 시퀀스 ≤ 2048 가정 |
| Gemma 4 E4B BF16 | 약 16GB | > 24GB 예상 | 미정 | 이번 범위 제외 |

전기 비용은 무시할 수 있다. 실제 비용은 사람의 검토 시간과 실패한 설치 시간이다. 별도 venv(`.venv-decision`)를 쓰고 기존 smoke용 `.venv`는 건드리지 않는다. 결정 학습 코드가 `AGPL-3.0-only`이므로 로컬 실험은 괜찮다. 다만 네트워크 서비스나 배포에 이 코드나 Studio Decision API를 포함하려면 라이선스를 따로 검토해야 한다. 이 경우 laya-train(Apache-2.0)이 대안이 될 수 있다.

## 7. 데이터·보정·평가 분리

- 기존 test는 이미 열람했으므로 새 적응형 설계의 최종 평가에 쓰지 않는다. **회귀 확인용**으로만 보고하고, 표에 "previously inspected"를 표시한다.
- 설정 선택에는 기존 validation을 쓴다. 선택 대상은 최대 4개 설정이다.
- 보정은 학습과 선택 모두에 쓰지 않은 별도 슬라이스로 한다. v3 생성 전에는 v2 train에서 scenario group 단위 10%를 떼어 내고, 그 그룹은 학습에서 완전히 제외해 보정 전용으로 둔다. 이렇게 떼어 낸 데이터는 더 이상 학습 데이터가 아니다. 보정 데이터를 validation과 섞지 않는다. v3가 생기면 v3의 calibration 분할로 대체한다.
- 최종 평가는 Codex의 v3 **새 scenario group, 사람 검토, 봉인된 test**로 한다. 선형 기준과 Laya 및 Gemma를 같은 행에서 end-to-end로 비교한다. 쌍대 비교(McNemar 또는 paired bootstrap 95% CI)에서 차이 구간이 0을 넘을 때만 "우수"라고 쓴다.
- v3 데이터 요청(Codex 소유):
  - 비Swallow missing-target `unsupported`(Grafana, korean_law, obsidian 각 서버)
  - 서버별 `clarify`와 `no_tool`
  - 후보 수 지름길 해소: 비선택 행도 7·8·96개 후보를 쓰고, 선택 행에도 4–6개 후보를 넣는다.
  - val과 test에 train 전용 도구 10개를 배치한다.
  - action enum을 분할별로 고르게 배치하고, 의도적인 unseen action 슬라이스를 둔다.
  - 개수는 강제 할당이 아니라 시나리오 그룹 단위 최소 수로 정한다. 예: 비선택 결정마다 그룹 ≥ 15, 서버마다 missing-target 그룹 ≥ 3.
  - 실제 로그는 이번 단계의 필수 조건이 아니다.

## 8. 최소 파일럿 설정과 게이트

모든 실행 공통: seed 3407, 13, 42 중 최종 설정만 3 seed로 실행한다. 실행 명령, revision, 패키지 버전, 데이터 해시, 피크 메모리, wall time을 기록한다. 학습은 train에서만 하고, 선택은 validation, 보정은 별도 슬라이스로 한다. test는 회귀 확인만 한다.

**G0 환경** (≤ 1시간): 새 venv에 unsloth와 unsloth-zoo `2026.10.6`, mlx `0.32.4`, torch를 설치한다(또는 `laya-train`). 설치된 `decision_mlx.py`와 `zoo/mlx/decision.py`를 위 커밋과 대조한다. `laya-multilingual@1720e3e3`를 로드하고 예제 1건으로 predict한다. 실패하면 laya-train MPS 경로로 전환한다. 둘 다 실패하면 Laya 파일럿을 중단하고 보고한다.

**G1 데이터·예산**: `build_dataset` 결과에서 skipped가 0이어야 한다. `options_collapsed`(alias가 잘려 같아진 옵션)도 0이어야 한다. 실제 tokenizer로 옵션 토큰 수와 state 잘림 비율을 보고하며, state 잘림은 5% 이하여야 한다. alias→ID 역변환과 prediction-v1 검증을 100% 통과해야 한다.

**G2 positive control**: train 32행으로 head-only를 거쳐 LoRA를 학습한다. ≤ 200 step 안에 train 정확도 ≥ 0.95가 나와야 한다. 실패하면 입력 표현이나 파이프라인 오류로 보고 중단한다.

**G3 자원**: Laya는 피크 ≤ 12GB, 1 epoch ≤ 30분, 추론 p50 ≤ 100ms(warm, shortlist 1창)를 목표로 한다. 측정값을 그대로 보고한다.

**G4 validation 비교**(같은 행, 선형 기준 대비):
- 후보 96개 행의 도구 정확도가 선형 이상이어야 한다.
- decision macro F1이 선형 이상이어야 한다.
- 비선택 recall이 하락하지 않아야 한다.
- 후보 밖 선택 0, shortlist 누락→unsupported 0이어야 한다.
- seen과 unseen action 정확도를 분리해 보고한다.
- 설정 3개 이내에서 통과하지 못하면 중단한다.

**G5 보정**: 별도 슬라이스 기준 ECE ≤ 0.10을 목표로 하되, 표본 수와 함께 보고한다.

Laya 권장 시작값:
- `max_len 1024`, `head_max_len 256`(학습 기본)
- LoRA r 16, alpha 16(기본 64와 비교 1회)
- micro-batch 8 × accum 2(유효 16)이면 epoch당 약 42 update. Studio 기본 8×8은 668행에서 epoch당 약 10 update라서 너무 적다.
- LoRA lr 2e-4, head lr 1e-4. 1 epoch head-only(사전학습 헤드 적응)를 먼저 한 뒤 LoRA를 2–4 epoch 학습한다.
- cosine 스케줄, weight decay 0.01, option shuffle

Gemma 단계는 Laya가 G0–G2를 통과한 뒤 별도 승인을 받아 진행한다.
- **사전 probe**: 4bit 로드 → 16행 → 20 step. 피크 ≤ 20GB, s/step을 측정한다. Qwen 템플릿이 Gemma에서 문제를 일으키지 않는지 정성 확인한다.
- **학습**: r 16, lr 1e-4, head lr 1e-4, micro 1 × accum 8, 시퀀스 ≤ 2048, 1–2 epoch, gradient checkpointing.
- **채택 조건**: G4에서 Laya와 선형을 같은 행 기준으로 넘을 것. Gemma를 진행하는 근거는 Laya가 train에서 underfit하는 경우뿐이다. Laya가 train에서 과적합하면 데이터 문제이므로 Gemma로 넘어가지 않는다.

## 9. 단계별 담당

| 단계 | 담당 | 산출물 |
|---|---|---|
| v3 데이터, 봉인 test, 보정 슬라이스 정의, venv와 실행기 | Codex | 데이터·계약·러너 |
| OOF shortlist, row→decision item 변환(alias·state), 창 확장 adapter, 보정, 평가와 비교 보고 | Claude | `src/jery/model/decision/`(가칭), `docs/model/` |
| v3 감사, 파일럿 오류 사례 뷰어(창별 결과 포함) | Antigravity | `apps/review/`, `docs/review/` |

## 10. 남은 불확실성 (명시)

1. PyPI `2026.10.6`이 위 main 커밋의 MLX 결정 학습을 포함하는지
2. M4 기본 GPU에서 Laya MLX 역전파 속도와 메모리. Apple 학습 수치는 어떤 출처에도 없다.
3. Gemma 4 E4B의 MLX Clef 학습 지원 여부(테스트 근거 없음, 멀티모달 로더 경유 여부), 사전 양자화 폴더 로드 가능 여부, Qwen 템플릿의 영향
4. mmBERT tokenizer로 센 실제 옵션과 state 토큰 수
5. 사전학습 Laya 헤드의 한국어 MCP 라우팅 zero-shot 수준. MASSIVE 한국어 0.49는 다른 과제다.
6. 668행 미검토 합성 데이터에서 신경망 모델의 이득이 선형 대비 통계적으로 구분될 만큼 클지

## 11. 다른 리뷰와의 합의·이견

독립 검토를 마친 뒤 Codex 메모(`tmp/codex-strategy-review.md`)와 Antigravity 초안(`main-2/docs/review/decision-data-strategy-review-20261017.md`, 미확정본)을 읽고 대조했다.

### 11.1 Codex 메모

**합의**:
- 선형 유지 → Laya 사전학습 헤드 적응 → FP16 base+LoRA 순서로 진행한다. Gemma 4bit는 조건부 비교군이고 BF16은 첫 시도 대상이 아니다.
- RLCD 없이 지도학습 soft CE만 쓰고, 커스텀 헤드는 포팅하지 않는다.
- C와 S를 분리하고, 누락이나 기술적 불확실성을 `clarify`나 `unsupported`로 매핑하지 않는다. 공개 계약은 유지한다.
- 독립된 창의 확률은 비교하지 않는다.
- 동적 alias를 쓰고, 정확한 Laya tokenizer로 측정하는 것을 게이트로 둔다.
- 대안 정답은 동등성 규약으로 균등 target을 쓴다. tool 단위 target에서는 같은 도구 ID를 먼저 중복 제거한다. action target은 선택된 정답 도구의 action으로만 만든다.
- 기술적 schema 실패는 `unsupported` 예시가 아니다.
- 위치 순열과 후보 수 맞춤은 모든 결정 유형에 적용한다.
- v2 test는 회귀 확인 전용으로만 쓴다. 검토된 group-disjoint 최종 test와 별도 보정 분할을 둔다.
- oracle shortlist는 진단용이다. 원 후보 C 기준으로 end-to-end 측정한다.
- M4 시간과 메모리 기준은 실제 step을 측정한 뒤 정한다.

**보완 및 이견**:
1. **k 선택**: Codex는 k ∈ {4, 8, 12} 측정을 제안한다. 내 측정에서 후보 96개 행의 선형 recall@5는 0.940으로 이미 @8(0.948)보다 낮다. k=4는 더 낮아 첫 창의 오선택 위험이 커지므로 후보에서 제외할 것을 제안한다. 대신 k=12(recall .966)를 Laya 옵션 예산(옵션당 15토큰)과 맞바꾸는 비교로 둔다.
2. **창 확장의 구체화**: Codex는 "widen 또는 delegate"까지만 정했다. 나는 내부 `none_of_these` 옵션과 창 단위 확장(§4)을 제안한다. 이 방식은 "gold는 C에만 의존" 원칙을 지키면서, 창에 정답이 없을 때 헤드가 멈출 수 있는 학습 신호를 준다. 음성 창은 train에서만 만들고 ablation으로 효과를 확인한다.
3. **"No calibration on training data"**: 같은 취지다. 다만 v3 전까지는 v2 train의 10% group을 학습에서 영구 제외해 보정 전용으로 쓰자고 제안한다. 이 데이터는 학습 데이터가 아니다.

### 11.2 Antigravity 초안: 이견과 권고 해결

| # | 초안 주장 | 문제 | 권고 해결 |
|---|---|---|---|
| A1 | Laya 지연 5–15ms, Gemma 결정 헤드 100–200ms, 학습 메모리 3.5GiB 또는 14.4GiB, OS·Hermes 6GiB 점유를 사실처럼 제시 | M4 측정이 없다. 14.4GB는 원문의 Nvidia VRAM 수치다. | 모두 [추정]으로 표시하고 G3 게이트에서 측정한다. |
| A2 | P_max < 0.40이나 높은 엔트로피면 확장하고, 그래도 모호하면 `clarify`로 폴백 | `clarify`는 요청의 성질이지 모델 불확실성이 아니다. 계약을 위반한다. | 낮은 확신에 따른 결정 변경은 금지한다. confidence는 출력으로만 전달한다. 확장은 `none_of_these`나 창 단위 `unsupported`로만 일어난다(§4). |
| A3 | 선형의 전체 96개 최대 로짓이 OOD 임계값 미만이면 `unsupported` 확정 | `unsupported`는 해당 사례의 후보 C 기준이지 전체 카탈로그 기준이 아니다. 선형 softmax 점수는 OOD 측정치로 보정되지 않았다. | `unsupported`는 C의 모든 창을 확인한 결과로만 낸다. 선형 점수는 순서로만 쓴다. |
| A4 | 1,000건 쿼터: select 700건, 도구당 최소 6건, Grafana 35% | Grafana 62개 × 6 = 372건이 700 × 35% = 245건을 넘어 동시에 만족할 수 없다. 도구당 val 1, test 1로는 도구별 지표가 의미가 없다. | 행 수 대신 scenario group 최소 수로 정한다. 예: 모든 도구가 train ≥ 2그룹, val·test 합산 ≥ 1그룹. 비선택 결정마다 그룹 ≥ 15. 서버마다 missing-target 그룹 ≥ 3. 서버 비율은 결과로만 보고한다. |
| A5 | 검토된 실제 Hermes 로그 50% 필수, fresh test에 실제 로그 포함 | 실제 로그 수집은 이번 단계에서 보류되었다. 필수 조건으로 두면 파일럿이 막힌다. | 지금은 실제 로그를 요구하지 않는다. v3 합성 데이터와 사람 검토로 진행하고, 실제 로그 평가셋은 별도 후속 단계로 둔다. 운영 정확도 주장은 그때까지 보류한다. |
| A6 | 기존 test가 "모델 sweep에 반복 노출" | 사실과 다르다. sweep과 선택은 validation으로 했고, test는 선택 후 1회 보고했다. 그 뒤 분석 과정에서 열람되었다. | "선택 후 보고되었고 이후 열람됨 → 새 적응형 설계에는 회귀 확인 전용"으로 정정한다. |
| A7 | AGPL 코드가 들어가면 "런타임 전체 소스 공개 의무", GATE-6 "Zero AGPL" | 과도하다. AGPL 의무는 수정본을 배포하거나 네트워크 서비스로 제공할 때 범위가 정해진다. 로컬 실험에는 문제가 없다. | 구성요소별 라이선스를 기록하고, 배포 결정 시점에 검토하는 것을 게이트로 둔다. 학습 경로 선택 근거로는 laya-train(Apache-2.0)이 대안으로 존재한다는 사실만 기록한다. |
| A8 | Gemma를 오프라인 교사로 써서 모호 사례 라벨링 보강 | 검토되지 않은 모델 라벨이 정답으로 들어갈 수 있다. 관측과 정답을 분리하는 원칙에 어긋난다. | 사람 검토용 제안으로만 쓰고 정답으로 승격하지 않는다. 이번 범위에서는 제외한다. |
| A9 | 8개 × 30자 ≈ 150–180토큰이면 예산에 완벽히 안착. 전체 MCP ID를 criteria 키로 사용 | 측정하지 않은 값이다. 전체 ID 키도 토큰을 쓴다. 의미 있는 차이를 30자로 표현하기 어렵다(예: Grafana Loki 계열 도구들). | 짧은 alias 키와 state에 넣는 압축 카드를 쓴다. 실제 tokenizer로 측정한다(G1). |
| A10 | `decision_type`과 `tool_routing` 두 질문 구조 | 비선택 행에는 tool 정답이 없어 질문이 건너뛰어진다. 결정과 도구가 따로 예측되어 서로 모순될 수 있다(select인데 도구 확률이 낮은 경우 등). | 단일 평면 `choice`(도구 + sentinel + `none_of_these`)를 기본으로 한다. 두 질문 구조는 ablation으로만 둔다. |
| A11 | 모든 사례에 항상 8개 후보를 강제 주입 | 원 후보가 4개인 사례에 후보를 추가하면 C가 바뀌어 정답 의미가 변한다. | 후보 수 지름길은 v3 데이터 생성(Codex)에서 해소한다. 모델 쪽에서 가짜 후보를 넣지 않는다. |
| A12 | GATE-1 ≥ 88%, GATE-2 ≥ 0.90, GATE-3 기존 "~10% 추정", GATE-5 ≤ 2GiB, ECE ≤ 0.05(보정 100건) | 절대 기준의 근거가 없고, 선형의 거짓 미지원 비율은 측정되지 않았다. 100건으로는 ECE 0.05를 통계적으로 판정할 수 없다. | 같은 검토 데이터에서 선형과 쌍대 비교하고 신뢰구간을 보고한다. 거짓 미지원은 선형도 같은 정의로 측정한다. 자원 기준은 G3 측정 후 정한다. ECE는 표본 수와 함께 보고한다. |
| A13 | 초안 §8.1 "Codex와 완전 일치" | 위 A2·A3·A5·A11은 Codex 원칙(불확실성 ≠ clarify, 계약 유지, 실제 로그 보류)과 충돌한다. | 합의 표는 정정이 끝난 뒤 다시 작성한다. |

**Antigravity와의 합의**: Laya 우선 진행. 점수 곱셈 금지. 조건부 확률의 분리 보정. 계층적 action 질의(스키마 enum을 옵션으로 렌더링)로 unseen action 문제 완화. any-of 평가. 위치 순열과 순서 불변성 확인(3회 순서 변경 추론). 세션과 시나리오 단위 격리. 초기 파일럿은 PyTorch MPS(laya-train)도 후보로 둔다. 이 마지막 항목은 내 G0 fallback과 같다.

### 11.3 최소 합의안 제안

1. 실제 로그는 이번 단계에서 요구하지 않는다. v3는 합성 데이터와 사람 검토로 만들고, scenario group 단위 최소 수를 정한다.
2. 보정 분할과 최종 test를 분리한다. 기존 v2 test는 회귀 확인 전용이다.
3. Laya 파일럿은 프로덕션 모델 선정이 아니다. G0–G5 게이트로 판정한다.
4. Gemma는 6건 생성 smoke 결과로 탈락시키지 않는다. Laya 이후 probe 게이트를 거쳐 비교한다.
5. 채택 판단은 같은 검토 데이터에서 원 후보 C 기준 end-to-end로 선형과 쌍대 비교한다.
6. 공개 계약은 prediction-v1을 유지한다. 내부 `none_of_these`와 창 확장은 adapter 내부 동작으로만 둔다.

## 12. 통합안 최종 판정 (2026-10-17)

대상: `<integration-root>/docs/data/decision-training-strategy.md`(통합 초안). 이번 판정에서 소스 재조사나 구현은 하지 않았다.

**판정: 수락(ACCEPT). 차단성 변경 사항 없음.**

다음 결정을 확인하고 동의한다.
- 공개 계약 prediction-v1은 변경하지 않고, 내부 형식 `decision-export-v1`을 둔다.
- §4의 `none_of_these`는 `window_none`으로 채택되었다. 원본 `unsupported`도 `window_none`으로 학습하고, C 전체를 확인한 경로에서만 공개 `unsupported`를 확정한다.
  - 이 방식이 내 원안(공개 `unsupported` 옵션과 내부 옵션을 둘 다 두는 안)보다 낫다. Laya 옵션이 k+3개로 하나 줄고(옵션당 `(256-16)//11 = 21`토큰), "gold는 C에만 의존" 원칙이 더 분명해진다.
- train 내부에서만 OOF 후보를 생성하고, dev·calibration·test에는 실제 후보 생성 결과를 사용한다.
- 실제 배포 조건 그대로 보정하고, 검색 누락 사례를 포함한다.
- 실제 로그를 선행 조건으로 두지 않는다.
- head-only 적응 후 FP16 LoRA로 진행한다.
- Gemma 4bit는 조건부로 두고, BF16은 연기한다.
- M4 성능은 모두 미측정으로 둔다.

**k 선택 권고**: 기본값은 **k=8**, 비교는 **k=12**(옵션 15개, 옵션당 16토큰)로 한다. k=4는 기본 운영 후보로 두지 않는다. 후보 96개 행에서 선형 recall@5가 이미 0.940으로 @8(0.948)보다 낮아, 첫 창의 오선택 위험만 커지기 때문이다. k=4는 k=8에서 G1 예산 측정으로 옵션이 충돌하거나 필수 정보가 잘릴 때에만 토큰 예산 진단용으로 측정한다.

**쿼터**: 1,000건 표는 장기적인 대표 사례 목표다. 32건 smoke 실행의 선행 조건이 아니다. fresh test의 도구당 1건은 커버리지 확인용일 뿐, 도구별 정확도의 근거가 아니다.

**문구 정리 요청(비차단)**:
1. 통합안의 "`4/8/12` 중 검증" 문구를 위 k 권고(기본 8, 비교 12, 4는 진단용)로 바꾼다.
2. 후보 96개 행에서는 창 하나당 음성 창이 최대 11개 생긴다. `window_none` 음성 창의 표집 비율(예: 양성 창 대비 상한)과 실제 개수를 export manifest에 기록하고, 이 비율을 ablation 대상으로 명시한다.
3. 복구 창 수에 상한을 두므로, 후보가 많은 원본 `unsupported` 사례는 대부분 위임으로 끝난다. 이 경우를 "unsupported recall"과 "위임 coverage"로 나눠 보고하라고 명시한다.
4. v2 기반 32건 smoke와 v2 파일럿은 v2에 calibration 분할이 없으므로 보정·품질 주장을 하지 않는다고 명시한다. "학습" 게이트에 positive control(작은 train에서 손실 감소와 train 정확도 상승)을 포함하는 것도 권한다.

**채택되지 않은 내 이전 제안(동의함)**:
- 공개 `unsupported` 옵션과 내부 옵션의 병존: 단일 `window_none`으로 대체되었다.
- v2 train의 10% group을 보정 전용 슬라이스로 분리: 향후 calibration 분할로 대체되었다.
- 기본으로 모든 창 순회: 상한 + 위임으로 대체되었다. 전체 순회는 최악 지연을 측정한 뒤 결정한다.
- Laya 시작값(micro 8 × accum 2, r16, lr 2e-4): micro 1부터 확인하고, rank 8/16과 lr 2e-5/2e-4를 작게 비교하는 방식으로 대체되었다.
- 그룹 단위 최소 수 쿼터: 대표 사례 4/1/1/1 목표로 대체되었다.
- §6 자원 추정표와 ECE ≤ 0.10, train ≥ 0.95 같은 수치 목표: 사실이나 기준으로 채택하지 않는다. probe 후 정한다.
- 최종 설정 3 seed: "고른 설정만 여러 seed"로 일반화되었다.

**남은 불확실성(파일럿 게이트로 유지)**:
- 실제 Laya tokenizer 기준 옵션과 state 토큰 수
- PyPI `2026.10.6`이 고정 소스 revision과 같은지
- M4에서의 Laya 역전파 시간과 메모리, 복구 창의 최악 지연
- Gemma 4 E4B Clef의 MLX 로드·역전파 가능 여부, 기존 4bit 폴더 사용 가능 여부, Qwen 템플릿의 영향
- 검토 데이터에서 선형 대비 이득이 통계적으로 확인될지
