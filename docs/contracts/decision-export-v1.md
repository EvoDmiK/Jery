# 결정 모델 파생 학습 형식 v1

## 전체 후보 범위 export

`scripts/build_laya_full_dataset.py`는 `export_scope=exhaustive_original_candidates`,
`purpose=full_synthetic_exploratory`를 명시해 원본 train 668·validation 248·test 246행을
모두 보존한다. 정답을 읽기 전에 후보 C를 shuffle하고 최대 8개씩 분할하며,
각 후보는 해당 원본 사례의 창들에 정확히 한 번 나온다. 창 수는 각각 4,172·1,592·1,577행이다.
원본 사례 수와 파생 학습 창 수를 혼용하지 않는다.

metadata에 `window_index`(0부터), `window_count`, `export_id`, `export_scope`를 더한다.
이 값들은 모델 입력이 아니다. 기존의 `derived_negative_window`는 파일럿에서 정답 도구를
제거해 만든 수작업 파생 음성 창을 나타내므로 전체 분할 export에서는 false다.
전체 분할 중 자연스럽게 정답 도구가 빠진 창도 `window_none` 정답을 갖지만 원본 결정은
여전히 `select_tools`다. 이 둘을 혼동해 원본 정답이 없다고 재해석하지 않는다.

학습에는 모든 창을 보존하고, 원본 사례 단위 평가는 모든 창의 후보 합집합이 C와 같은지
확인한 후 수행한다. 창 간 softmax 비교는 사용하지 않는다. 다수결·결승 집계와 내부
delegate 처리는 별도 평가 규칙이며 공개 prediction-v1에 결정을 추가하지 않는다.
delegate는 정확도에서 오답으로 세고 coverage를 별도로 보고한다. test는 기존 실험에서도
관찰한 회귀 평가 분할이므로 독립적인 새 성능 증거로 주장하지 않는다.

`decision-export-v1`은 기존 `mcp-routing-v2`에서 만드는 별도 파생 형식이다.
원본 v2 라벨과 [공개 prediction-v1](prediction-v1.md)의 의미는 바꾸지 않는다.
첫 구현은 Laya의 공학 파일럿이며 의미적 성능 평가용 데이터셋이 아니다.

## 행과 모델 경계

각 JSONL 행은 `state`, `questions`, `gold`, `metadata`를 가진다.

- `state`: `request`, 호출 전 `context`, 현재 창의 압축 `candidates` 카드.
- `questions.route_in_window`: `type=choice`, 처리 지침과 `criteria`.
- `gold.route_in_window`: 대표 `label`과 허용 선택지의 `probabilities`.
- `metadata`: 원본 사례·시나리오·분할·카탈로그·원 후보·창·alias 대응·원본 정답과 검토 상태.

추론 입력은 state와 questions뿐이다. gold는 손실 계산에만 쓰고 metadata는
계보·감사·평가에만 사용한다. source ID, 시나리오 ID, split, 검토·실행 상태와
원본 정답을 입력 문자열에 포함하지 않는다. 원본 요청의 텍스트는 데이터로 취급한다.

카드는 `alias`, `server`, `name`, `summary`만 가진다. summary는 tools.json의
description에서 공백을 정리한 뒤 첫 120자를 사용하는 초기 표현이다. 정답을 보고
설명을 만들지 않는다. 이 글자 수 제한은 토큰 예산이나 의미 보존을 보장하지 않으므로
실제 tokenizer로 입력 손실을 검사하고 본 학습 전 표현 품질을 검토해야 한다.

## 원 후보와 내부 창

원래 허용 후보 C와 내부 창 W를 분리한다. W는 C의 부분집합이며 가짜 도구를
추가하지 않는다. `T0` 등 창별 alias는 원래 도구 ID와 일대일로 대응한다.
도구 선택지 설명은 짧은 alias로 하고, 명세 정보는 연결된 state 카드에 둔다.

내부 선택지는 현재 창의 도구 alias와 `no_tool`, `clarify`, `window_none`이다.
`window_none`은 공개 결정이 아니며 현재 창 안에 적합한 도구가 없다는 뜻이다.

| 원본 의미 | 파생 정답 |
|---|---|
| no_tool / clarify | 같은 문자열 |
| select_tools, 창 안에 허용 도구가 있음 | 해당 alias들 |
| select_tools, 원 후보에는 정답이 있으나 창에는 없음 | window_none |
| 원 후보 C 기준 unsupported | window_none |

복수 대안은 동일 도구 ID를 먼저 중복 제거하고 창에 있는 서로 다른 허용 도구에
동일 질량을 준다. 합은 1이다. 대표 label은 그중 하나이며 분포와 동등성 규약을
함께 보존한다. 이 값은 사람이 측정한 확률이 아니다. action은 첫 export에서
모델링하지 않지만 metadata의 원본 acceptable_choices에서 보존한다.

원래 선택 사례의 음성 창은 모든 허용 도구를 C에서 제거하여 만든다. 원본
decision은 여전히 select_tools로 남긴다. 내부 window_none을 그대로 공개
unsupported로 변환하지 않는다. 추가 창 검토·원 후보 위임을 거쳐야 하며,
현재 창만으로 지원 불가를 확정할 수 없다.

## 첫 파일럿 구성과 한계

`scripts/build_laya_pilot.py`가 기존 train에서 32행, validation에서 12행을 만든다.
시나리오 그룹은 각각 26개와 10개이며 분할 사이 교집합이 없다. 각 결정 유형에서
서로 다른 시나리오를 우선하되 부족한 no_tool·clarify는 같은 분할의 표현 변형을
사용한다. test는 읽지 않는다.

이 export는 후보가 최대 8개인 원본 행을 사용한다. train 4행, validation 2행은
선택 정답을 창에서 뺀 파생 음성 창이다. OOF 검색기의 후보도 아니고, 후보 개수
지름길을 해결한 데이터도 아니다. 모든 사례의 synthetic·needs_human_review와
execution_verified=false 상태는 원본대로 유지한다. 정확도·보정·운영 채택의
근거로 쓰지 않는다.

출력 디렉터리는 새로 만들어야 하며 기존 export는 덮어쓰지 않는다. manifest에
입력·출력 SHA-256, seed, 버전, 분할·그룹·결정·음성 창 건수와 한계를 기록한다.
표현이나 후보 변환을 바꾸면 새 출력 경로와 해시로 구분한다.

```sh
python3 scripts/validate_mcp_dataset.py
python3 scripts/build_laya_pilot.py --output outputs/laya-pilot-20261017/export
python3 -m unittest discover -s tests/data -p test_laya_export.py
```
