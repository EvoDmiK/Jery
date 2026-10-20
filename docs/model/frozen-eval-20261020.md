# 고정 10개 도구 frozen 평가 gate (frozen-eval-v1, 2026-10-20)

신규 실제 요청 평가셋(약 100개, 도구별 10개)을 두 고정 Laya head로 **한 번** 채점하기 위한 읽기 전용 gate·runner다.
`src/jery/evaluation/frozen.py`, 고정 사양은 `configs/model/frozen-eval-ten-tools-20261020.json`이다.
평가셋은 아직 없다. 이 코드로 학습·튜닝·하이퍼파라미터 탐색을 하지 않으며, root가 freeze를 승인하기 전에는
모델 예측을 만들지 않는다.

## 비교 대상(고정)

| 이름 | MLflow Child | 비고 |
|---|---|---|
| `baseline-regularized` | `ca0d15030d6543a99ba6a7084679d6bc` | 정규화 탐색 선정 head. **v1 짧은 설명**(`hermes-ten-tools-b4648cc19a90`)으로 학습 |
| `augmented-train240` | `5ee0cebebfbd465898abdd4b5f300a56` | v2.1 설명 + train 240 |

두 head 모두 같은 base(`models/laya-multilingual`, sha `9d628f…f204`), 같은 v2.1 criteria
(`catalog-grounded-criteria-v2.1-native-budget`, catalog `hermes-ten-tools-2289142ab57a`), 같은 instruction,
같은 후보 순서 seed와 같은 native 입력 빌더로 채점한다. baseline은 학습 때와 다른 설명을 보므로 이 비교는
"v2.1 입력 조건에서의 두 head"이며 설명 변경 효과와 데이터 보강 효과를 분리하지 않는다.

## 입력 계약(root 승인, 2026-10-20)

frozen 폴더에 다음이 있어야 한다. root와 AGY가 manifest를 이 형식에 맞춘다. 필드가 바뀌면 버전을 올린다.

- `manifest.json`: `schema_version=frozen-eval-v1`, `status`(`candidate|frozen`), `catalog_version`,
  정수 `candidate_order_seed`, `leak_audit={status, file}`, `sha256`(상대 경로 → SHA-256; `cases.jsonl`,
  `tools.json`, provisional 모드면 `ai-proposals.json` 포함, 나열된 파일은 모두 검사).
- `leak_audit.file`(예: `leak-audit.json`): `sha256`에 포함되고 내용의 `status=passed`여야 채점한다. 자기 신고 플래그만으로는 안 된다.
- `tools.json`: `datasets/hermes-ten-tools-v2/tools.json`과 바이트 동일(sha `11c0253a…3847`).
- `cases.jsonl`: `id`, `scenario_id`, `request`, `context`(결정 이전 메시지만), `candidate_tools`(고정 10개),
  `tool_catalog_version`, `decision`, `acceptable_choices`, `source.kind=real`, `review_status`, `execution_verified`.
- `ai-proposals.json`(선택): `{case_id: [{proposer, disposition, decision, acceptable_choices, confidence, rationale}]}`.

## 거부 조건

모델을 부르기 전에 하나라도 걸리면 `GateError`로 멈춘다(일부 사례만 버리고 채점하지 않는다).

- 빈 사례 파일, manifest schema·catalog 불일치, 채점 시 해시된 leak audit 파일 미통과, 파일 해시 변경(입력·정답·카탈로그), 고정 카탈로그·criteria 변경
- 채점 시 `status!=frozen` 또는 `--authorize-manifest-sha256`가 manifest 해시와 다름
- 후보가 정확히 고정 10개가 아니거나 카탈로그 밖 도구, `source.kind!=real`
- request/context에 `mcp__` 도구 ID 노출, 정규화 입력 중복, `--exclude-export`로 준 학습·검증·테스트 export 입력과 중복
- 미해결 정답: `human` 모드는 `review_status=human_reviewed`이고 `select_tools`·서로 다른 도구 1개·유효 action이어야 한다.
  `provisional-ai` 모드는 서로 다른 proposer 2명 이상이 모두 `disposition=label`로 같은 단일 도구를 제안해야 한다.
  다중 도구·모호 사례를 강제로 단일 라벨로 만들지 않는다.

모델 입력은 `model_input`을 거친 request·context와 고정 criteria뿐이다. 사례 ID, 정답, 출처, 분할, 검토 상태는
입력에 들어가지 않는다(정답·출처를 바꿔도 scorer가 보는 입력이 같음을 테스트로 확인).

## 출력과 라벨 표시

`--out`(새 폴더)에 `gate.json`, 후보별 `<name>.cases.jsonl`(id, gold_tool, predicted_tool, correct, 도구별 확률)과
`<name>.metrics.json`(학습과 같은 `case_accuracy`, `tool_macro_f1`, `per_tool`, window soft CE, 시간, run ID, head·base 해시)을 쓴다.
`gate.json`과 metrics에는 `missing_by_tool`(도구별 목표 10개 대비 부족분), `meets_target`, `review_status_counts`,
`leak_audit_ready`를 남긴다. 목표에 못 미치면 claim에 부분 집합임을 적고 real100으로 표시하지 않는다.
provisional 모드는 사례의 `review_status`(예: `needs_human_review`)를 per-case 출력에 그대로 보존한다.
`label_basis`는 `human_reviewed` 또는 `provisional_ai_agreement (not human reviewed; exploratory only)`이고, provisional
결과는 실제 사용 정확도·사람 검토 정답 성능으로 보고하지 않는다. AI 제안은 human review로 승격하지 않는다.

```sh
# freeze 전·후 언제든: 모델 없이 gate만
PYTHONPATH=src python3 -m jery.evaluation.frozen --frozen <dir> --labels human --gate-only \
  --exclude-export outputs/laya-ten-tools-contrast-20261020/augmented \
  --exclude-export outputs/laya-ten-tools-20261020/full
# root freeze 승인 후 1회(CUDA 서버)
PYTHONPATH=src python -m jery.evaluation.frozen --frozen <dir> --labels human \
  --authorize-manifest-sha256 <manifest sha> \
  --head ca0d15030d6543a99ba6a7084679d6bc=<trainable.safetensors> \
  --head 5ee0cebebfbd465898abdd4b5f300a56=<trainable.safetensors> --out <새 폴더>
PYTHONPATH=src python3 -m unittest tests/evaluation/test_frozen.py
```

## 한계

- 실제 평가셋·라벨이 없어 벤치마크는 실행하지 않았다. 테스트는 가짜 scorer로 gate와 지표 경로만 확인한다.
- CUDA 경로(`load_cuda_model`→`restore_head`→`CudaScorer`)는 기존 학습 runner와 같은 함수지만 이 CLI로는 아직 실행하지 않았다.
  head 경로는 root가 MLflow artifact에서 지정하고 SHA-256을 결과에 기록한다.
- 누수 검사는 정확·정규화 일치만 본다. 근접 의미 중복은 root의 leak audit에 의존한다.
- 약 100개 단일 실행이므로 두 head 차이는 신뢰구간 없이 해석하지 않는다. MLflow Parent/Child 기록은 root가 맡는다.
