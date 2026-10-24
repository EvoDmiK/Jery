# Hermes 고정 10개 도구 v4

학습 데이터를 600개에서 **800개(도구당 80개)**로 늘리고, 도구 설명에 혼동하기 쉬운 작업의 경계를 명시했다. 기존 600개의 요청·맥락·정답·선택지 순서는 보존했다. 추가 200개는 [합성 원문 v3](../hermes-ten-tools-contrast-source-v3/README.md)에 있다.

## 비교용 구성

| MLflow profile | Train | Validation | Test | 도구 설명 |
|---|---:|---:|---:|---|
| baseline | 600 | 20 | 20 | 기존 |
| description_only | 600 | 20 | 20 | 개선 |
| data_only | 800 | 20 | 20 | 기존 |
| combined | 800 | 20 | 20 | 개선 |

기존 설명 구성의 validation/test는 원본과 바이트까지 동일하다. 개선 설명 구성은 요청·맥락·정답·alias·scenario를 보존하면서 criteria와 카탈로그 버전만 변경했다. 이미 관찰한 validation/test이므로 회귀·개발 검증용이다. 독립 실제 test200을 완성한 데이터가 아니다.

원래 도구 ID·description·input_schema는 보존한다. 새 카탈로그의 `routing_description`은 모델용 짧은 한국어 설명이고, export의 `questions.route_in_window.criteria`와 일치한다. 문서 읽기와 수정, 조회와 주문, 과거 가격과 현재가, DQL과 .base 작업을 구분한다. `publish_routing_descriptions=true`인 새 criteria에서만 이 필드를 추가하므로 기존 버전의 재생성 해시를 바꾸지 않는다.

새 catalog는 `hermes-ten-tools-0b320e20bf3f`, 표현은 `catalog-grounded-boundary-criteria-v3`, export 계약은 기존 `decision-export-v2`다. 라벨 의미·후보 수·도구 action/schema는 바꾸지 않았다. 이전 카탈로그와의 마이그레이션은 같은 사례 ID/alias에 새 criteria를 연결하고 metadata.tool_catalog_version을 갱신하는 방식이다. 모델 추론에서도 학습 당시 카탈로그를 함께 사용해야 한다.

## 검증

- 최종 학습800, 도구당80, 전체475개 시나리오. 신규200개는120개 시나리오·24개 문맥 사례다.
- 기존 데이터 보존, 재생성 결정성, 도구/action 참조, 후보·gold 매핑, 분할 간 중복·시나리오 겹침 검사를 통과했다.
- 실제 Laya tokenizer/native builder에서 네 구성 총 2,960개 입력을 검사했다. 건너뜀·잘림·특징 불일치·정답 매핑 불일치 0건이다. 최대 sequence는 개선 설명382, 기존 설명358 tokens이며 max_len1024/head_max_len256을 유지했다.
- 초안의 긴 설명은 후보별24-token 한도를 초과해 거부됐다. 최종 설명을 줄여 통과시켰고 길이 제한을 완화하지 않았다.
- 새 학습이나 모델 성능 평가는 실행하지 않았다. 검증 통과는 형식·보존 검증이며 정확도 향상 증거가 아니다.

`native-format-audit.json`과 원문 폴더의 `audit.json`에 근거를 기록했다. 모든 신규 정답은 사람 검토 대기다. 서로 다른 라벨의 대조 쌍은 같은 scenario로 묶여 있으므로 기존 homogeneous-only CV 생성기를 새 pool에 직접 사용할 수 없다. 재학습 전 group을 보존하는 CV 계획이 필요하다.

## 재현

기존 v3 export가 없는 checkout에서는 v1→v2→v3 README 순서로 먼저 재생성한다. 아래 기존 빌더를 두 번 호출하면 네 구성이 만들어진다. 이미 존재하는 출력 경로는 덮어쓰지 않고 거부한다.

```sh
python3 scripts/build_ten_tool_contrast.py \
  --base outputs/laya-ten-tools-training-v3-20261020/final-exports/augmented \
  --catalog-dir datasets/hermes-ten-tools-v3 \
  --authored datasets/hermes-ten-tools-contrast-source-v3/cases.jsonl \
  --criteria configs/training/laya-ten-tools-criteria-v3-20261024.json \
  --dataset-version hermes-ten-tools-v4 \
  --dataset outputs/reproduce-v4/catalog \
  --output outputs/reproduce-v4/combined-controls

python3 scripts/build_ten_tool_contrast.py \
  --base outputs/laya-ten-tools-training-v3-20261020/final-exports/augmented \
  --catalog-dir datasets/hermes-ten-tools-v3 \
  --authored datasets/hermes-ten-tools-contrast-source-v3/cases.jsonl \
  --criteria configs/training/laya-ten-tools-criteria-v2-20261020.json \
  --dataset-version hermes-ten-tools-v4-data-control \
  --dataset outputs/reproduce-v4/data-catalog \
  --output outputs/reproduce-v4/data-controls

PYTHONPATH=src:. python3 -m unittest discover -s tests/data -p test_ten_tool_contrast.py
```

각 출력의 `description_control`은 기존600개, `augmented`는800개다. 실제 export는 `outputs/laya-ten-tools-v4-20261024/{data-controls,combined-controls}/`에 있고, Git에는 원문·criteria·카탈로그·manifest·검증 집계만 보관한다. MLflow에는 JSON 배열과 바로 읽을 수 있는 JSONL ZIP을 모두 저장한다.
