# 10개 도구 학습 보강 원문 v3 — Claude 작성분

`configs/model/training-source-v3-claude.jsonl`은 Claude(claude-opus-5-5)가 손으로 작성한 한국어 합성 **train 전용** 원문 180개다. 평가셋이 아니며, 사람이 검토한 정답이나 실행 로그도 아니다. 루트(dev-codex)가 수용한 사례를 `datasets/`로 옮기고 이 중간 파일은 통합 전에 제거할 수 있다.

- 기준 커밋: 루트 baseline `659de08`(통합 대상 `dev-codex`). 이 브랜치에는 병합하지 않고 `git show`로 읽었다.
- 참조한 파일: 카탈로그 `datasets/hermes-ten-tools-v2/tools.json`(`hermes-ten-tools-2289142ab57a`), 기존 train 원문 `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl`, 기준 `configs/training/laya-ten-tools-criteria-v2-20261020.json`. validation/test 요청은 읽지 않았다.
- 파일 sha256: `9f6311adfd8bf0211f06d8b81876397b53051b4f887725c3840c8eb55a94a514`

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
