# 10개 도구 학습 보강 원문 v3 — Claude 작성분

`configs/model/training-source-v3-claude.jsonl`은 Claude(claude-opus-5-5)가 손으로 작성한 한국어 합성 **train 전용** 원문 180개다. 평가셋이 아니며, 사람이 검토한 정답이나 실행 로그도 아니다. 루트(dev-codex)가 수용한 사례를 `datasets/`로 옮기고 이 중간 파일은 통합 전에 제거할 수 있다.

- 기준 커밋: 루트 baseline `659de08`(통합 대상 `dev-codex`). 이 브랜치에는 병합하지 않고 `git show`로 읽었다.
- 참조한 파일: 카탈로그 `datasets/hermes-ten-tools-v2/tools.json`(`hermes-ten-tools-2289142ab57a`), 기존 train 원문 `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl`, 기준 `configs/training/laya-ten-tools-criteria-v2-20261020.json`. validation/test 요청은 읽지 않았다.
- 파일 sha256: `fb557f56b94ba781cd4ade7f6167df7f1898f2d0234b1e0e0e6fb13e2edaa043`(커밋 `2e787a9`). 최초 버전 `9f6311ad…a514`와 비교하면 `train-v3.claude.swallow_get_stock_price.09.0`의 맥락·요청만 다르다. 라벨, ID, scenario_id는 같다.

## 스키마

기존 원문과 같다: `{id, scenario_id, split:'train', request, context, tool, action, source, review_status:'needs_human_review', execution_verified:false}`. ID·scenario_id 접두어는 `train-v3.claude.<도구>.<가족>`이다. `source`에는 `kind:'synthetic'`, `author`, `model`, `method`, `form`(direct·implicit·context), `intended_contrast`, `rationale`이 있으며 검토용일 뿐 모델 특징이 아니다. 모델 입력은 `request`와 `context`뿐이다.

## 구성

| 도구 | 사례 | 시나리오 가족 | 맥락 사례 | action |
|---|---|---|---|---|
| `vault` | 18 | 14 | 3 | search 4, read 2, create 2, rename 2, list 1, move 1, copy 1, delete 1, split 1, combine 1, concatenate 1, fragments 1 |
| `edit` | 18 | 14 | 3 | patch 7, window 5, append 4, at_line 2 |
| `graph` | 18 | 14 | 2 | backlinks 3, neighbors 3, path 2, forwardlinks 2, statistics 2, traverse 1, tag-analysis 1, shared-tags 1, search-traverse 1, tag-traverse 1, advanced-traverse 1 |
| `dataview` | 18 | 14 | 3 | query 10, metadata 3, validate 2, list 2, status 1 |
| `bases` | 18 | 14 | 3 | view 4, export 4, read 3, query 3, list 2, create 2 |
| `swallow_get_stock_price` | 18 | 14 | 2 | null |
| `swallow_get_price_history` | 18 | 14 | 2 | null |
| `swallow_get_portfolio` | 18 | 14 | 2 | null |
| `swallow_order` | 18 | 14 | 2 | null |
| `swallow_backtest` | 18 | 14 | 2 | null |

도구당 14개 가족 중 4개만 형제 2개(바꿔 쓴 요청 또는 맥락 해결형)를 두고 나머지는 단일 사례다. 형식은 direct 100, implicit 56, context 24개다. 금융 사례는 30개 이상 국내 종목, 5~1000일 기간, 토스·네이버·교차검증 소스, RSI 기준·익절·손절·초기자금 조합을 나눠 쓴다.

## 작성 원칙

- 단일 도구 `select_tools`만 넣었다. clarify·unsupported·여러 단계 요청은 넣지 않았다.
- action은 카탈로그 enum 값만 쓴다. Swallow는 null이다. `bases`의 `evaluate`처럼 설명에만 있고 enum에 없는 기능, 주문 수량처럼 스키마에 없는 인자는 쓰지 않았다. 주문 요청에는 필수 인자인 종목명·코드·가격·매수/매도를 모두 넣었다.
- 맥락은 호출 전 user/assistant 메시지뿐이며, 요청과 합치면 대상과 인자가 확정된다. assistant 메시지는 도구 이름이나 정답을 말하지 않는다.
- 일부러 넣은 대조: 본문 검색·읽기(vault) / 본문 수정(edit) / 링크 탐색(graph), DQL·인라인 필드(dataview) / `.base` 파일(bases), 현재가·단일 보유(stock_price) / 일별 종가(price_history) / 계좌 전체(portfolio) / 전략 검증(backtest) / 주문(order).
- 파일 경로·인물 이름은 가상이며 실제 개인정보나 비공개 Hermes 자료는 없다. 종목명과 코드는 공개 상장 정보다.

## 검증 (재현)

baseline 빌더 `scripts/build_ten_tool_contrast.py`(659de08)를 고치지 않고 임시 경로로 실행했다. 기존 v1 원문과 이 파일을 이어 붙인 360개를 `--authored`로 넘겼고, `--base`는 기존 `outputs/laya-ten-tools-20261020/full` export다.

```sh
python3 scripts/build_ten_tool_contrast.py --base outputs/laya-ten-tools-20261020/full \
  --catalog-dir datasets/hermes-ten-tools-v1 --authored <v1+v3 결합 파일> \
  --criteria configs/training/laya-ten-tools-criteria-v2-20261020.json \
  --dataset <임시 경로> --output <임시 경로>
```

결과: augmented train 420행(기존 60 + v1 180 + v3 180), validation/test 각 20행은 그대로다. 빌더 검사를 모두 통과했다. 검사 항목은 다음과 같다.

- action enum과 Swallow null
- 도구 ID 누출
- 기존 train·held-out·v1과 정규화한 입력의 중복
- 분할 간 scenario 그룹 겹침
- 도구 균형
- `needs_human_review`/`execution_verified=false` 유지

추가로 v1 원문과의 요청 유사도(difflib)를 확인했다. 최대 0.71로 문장을 복사한 사례는 없었다. 데이터셋, 가중치, held-out, 하이퍼파라미터는 바꾸지 않았고 학습도 실행하지 않았다.

## 한계

모든 사례는 `needs_human_review`다. 다른 작성자의 AI 교차 검토를 거쳐도 `human_reviewed`가 되지 않는다. `vault` concatenate와 `edit` append, `edit` window와 `vault` update처럼 경계가 미묘한 사례는 `source.rationale`에 판단 근거를 적었으니 사람 검토에서 확인해야 한다.

## AGY180 교차 검토와 빌더 cad7a3d 점검 (Claude, AI 검토)

사람 검토가 아니다. AGY 사례는 모두 `needs_human_review`로 남고 실행 검증도 하지 않았다. validation/test 문장은 작성이나 판정에 쓰지 않았다.

### 빌더 `cad7a3d` (`scripts/build_ten_tool_contrast.py`, `tests/data/test_ten_tool_contrast.py`)

임시 detached worktree에서 읽기만 했다. 브랜치 병합이나 스크립트 수정은 하지 않았다.

- 기존 테스트 5개는 모두 통과했다(`python3 -m unittest tests.data.test_ten_tool_contrast`).
- 실제 v2 augmented export(train 240)에 Claude180과 AGY180을 넣어 확장 모드로 빌드했다. 결과는 다음과 같다.
  - train 600행, 도구당 60개
  - validation/test는 바이트 동일
  - train 앞부분 240행은 바이트 그대로 유지
  - 새 행마다 gold alias→tool, `acceptable_choices` tool/action, state, scenario_id, `needs_human_review`/`execution_verified=false`가 원문과 같음
  - 원본 export 파일은 바뀌지 않음
- 음성 가드 17개는 모두 출력 경로를 만들기 전에 거부했다: held-out 복사, 기존 train 정규화 복사, 기존 source_id 재사용, 신규 ID 중복, held-out scenario_id, enum 밖 action(`evaluate`), Swallow action 비null, `human_reviewed` 승격, `execution_verified=true`, 비합성 출처, 도구 ID 누출, `tool` 역할 맥락, 빈 요청, 불균형, 모르는 도구, train이 아닌 split, base criteria 변조.
- 정확성 결함은 찾지 못했다. 참고할 점은 세 가지다.
  - 확장 모드의 `description_control` profile은 base(v2 augmented 240)를 그대로 복사하므로 이름과 내용이 다르다.
  - 도구 ID 누출 검사는 `mcp__…` 전체 ID만 찾는다.
  - 필드가 빠지면 ValueError 대신 KeyError가 난다. 다만 쓰기 전에 실패한다.

### AGY180 판정

- 입력: `outputs/laya-ten-tools-training-v3-20261020/agy-author-snapshot.jsonl`, sha256 `beb6750f02063e9688a3e6133eb198fa7c832fb2424f9cbc843fd33f7f61b49e`
- 결과 파일: `configs/model/training-v3-agy-review.json`(사례별 `{id, reviewer, status, expected_tool, expected_action, rationale}`)
- 180개를 모두 봤다. accept 152, revise 28, exclude 0이다. 도구·action 라벨이 틀린 사례는 없었다.

revise 사유:

| 사유 | 건수 |
|---|---|
| assistant가 정답 동작을 예/아니오 확인으로 말함 | 11 |
| 종목·숫자만 바꾼 같은 문장 틀 또는 같은 의미의 다른 scenario | 11 |
| 요청에 action 이름이 괄호로 들어감 | 3 |
| `from_buffer`를 맥락 없이 사용 | 1 |
| assistant 맥락이 실행 결과처럼 파일 존재를 단정 | 1 |
| 경로 앞 공백 | 1 |

정확한 ID와 수정안은 결과 JSON과 루트 메시지에 있다. 그 밖에 AGY 요청의 경향으로 "Swallow" 접두어 37건, "옵시디언" 23건, "~말고" 대조어 34건이 있다. 표면 단서에 기댈 위험이 있으니 사람 검토 때 함께 볼 것을 권한다.

자체 점검과 수정: 내 원문 `train-v3.claude.swallow_get_stock_price.09.0`도 assistant가 "주가를 확인해 드릴까요?"라고 확인하는 패턴이었다. 루트 요청에 따라 다음과 같이 고쳤다(`2e787a9`).

- user: "크래프톤(259960) 실적 발표가 오늘이라던데."
- assistant: "그렇군요. 어떤 걸 도와드릴까요?"
- 요청: "그 종목 지금 시세랑 오늘 등락 좀 보여줘."

빌더 확장 모드에서 v2 train 240과 held-out 대비 정규화 중복이 없었다. 기존 train 240과의 최대 유사도는 0.26이다.

### AGY 수정본 최종 재검토 (Claude, AI 검토)

- 입력: `outputs/laya-ten-tools-training-v3-20261020/agy-revised-snapshot.jsonl`, sha256 `8d44816f8543f354e4be27c074d0700d156a0e02428fc18299de0ac7cd24e715`
- 첫 snapshot(`beb6750f…b49e`)과 ID별로 대조했다. 추가·삭제된 ID는 없다. 바뀐 77행의 구성은 다음과 같다.
  - 첫 판정 revise 28행: 모두 고쳐졌다.
  - 그 밖에 문장이 바뀐 41행: 예/아니오 선택지형 assistant를 중립 질문으로 바꾸는 등. 모두 다시 읽었다.
  - source 메모만 바뀐 8행
- 바이트가 같은 103행은 첫 판정 accept를 다시 읽고 유지했다.
- 라벨이 바뀐 행은 `obsidian_vault__edit.06.0` 하나다. 맥락 없는 from_buffer를 경로·문구가 명시된 window 요청으로 다시 썼다.
- 최종 판정: accept 180, revise 0, exclude 0. 첫 판정은 `configs/model/training-v3-agy-review.json`의 `history`와 사례별 `initial`에 남겼다.
- 실제 검사(cad7a3d 빌더, Claude `fb557f56` + AGY `8d44816f`):
  - train 600행, 도구당 60개
  - validation/test는 바이트 동일, 앞 240행 유지
  - 새 행마다 라벨·state·scenario 보존
  - 음성 가드 17/17 거부
  - 도구당 시나리오 가족 12개 이상, 형제는 최대 2개
  - 같은 도구 안 다른 scenario 사이에서 마스킹 difflib 0.80 이상 쌍 0개
  - 정답을 확인하는 assistant 0건
- 남은 관찰(차단 아님): AGY assistant 맥락 30개 중 21개가 "…에 대해 어떤 작업을 할까요?"로 거의 같다. 도구와 무관한 문장이라 라벨 단서는 아니지만 다양성이 낮다. "~말고" 대조어는 37건, "Swallow" 접두어는 21건이다. 사람 검토 때 표면 단서 위험을 함께 볼 것을 권한다.
