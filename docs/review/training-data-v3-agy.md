# 고정 10개 도구 학습 보강 데이터 v3 (`training-source-v3-agy.jsonl`) 작성 및 검증 보고서

## 1. 개요 및 산출물 무결성

`datasets/hermes-ten-tools-v2/tools.json`의 고정 10개 도구 명세(`input_schema`)와 `configs/training/laya-ten-tools-criteria-v2-20261020.json`의 선택지 기준을 바탕으로, 학습 분할(`split="train"`) 전용 한국어 합성 사례 **180건**(도구당 정확히 18건)을 작성하고 `scripts/build_ten_tool_contrast.py` 및 `apps/review/frozen_eval_audit.py` 헬퍼로 검증했다.

- **작성 파일**: `apps/review/training-source-v3-agy.jsonl`
  - **행 수**: `180`행 (10개 도구 × 18행)
  - **고유 시나리오 그룹(`scenario_id`)**: `121`개 (`vault` 13개, 나머지 9개 도구 각 12개; 도구당 최소 12개 시나리오 기준 충족, 시나리오당 최대 2개 형제 변형)
  - **ID / 시나리오 접두사**: `train-v3.agy.<short_tool>.<family_idx>` / `train-v3.agy.<short_tool>.<family_idx>.<variant_idx>`
  - **SHA-256**: `d3a3aa81cc33c80918fbf7f1d818b60a55ea29991a1788619e76c8bd64834cf3`
- **참조 기준 파일 해시 (변경 없음)**:
  - `datasets/hermes-ten-tools-v2/tools.json`: `11c0253a5de892afaec5836b847cfb70bc76c4caf23e96b8b41a414183123847` (`catalog_version=hermes-ten-tools-2289142ab57a`)
  - `configs/training/laya-ten-tools-criteria-v2-20261020.json`: `2289142ab57a7b2f156cf19eeaca0b8d064f90c03d28bb6c19723045b162438d` (`seed=3407`)
  - `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl`: `a9650b720f1faa7f57fb61343e69a88222486e1bffb35dc6f0878c5b5c86954f`

---

## 2. 도구별 건수 균형, 시나리오 다양성 및 카탈로그 Action 검증

모든 사례는 단일 도구 선택(`select_tools`) 과제로 구성되었으며, `clarify`·`unsupported`·다단계(lookup 후 주문 등) 사례는 포함하지 않았다. 특히 `bases` 도구의 경우 설명문(prose)에 언급된 `evaluate`가 실제 `input_schema.properties.action.enum`에는 없으므로 `input_schema`의 6개 enum(`list`, `read`, `create`, `query`, `view`, `export`)만 엄격히 사용했다.

| 도구 ID | 사례 수 | 고유 시나리오 가족 수 | 시나리오당 최대 형제 수 | Action 분포 (`input_schema` 대조 결과) |
|---|---:|---:|---:|---|
| `mcp__obsidian_vault__vault` | 18 | 13 | 2 | `list`(2), `read`(2), `create`(2), `update`(2), `fragments`(2), `delete`(1), `search`(1), `move`(1), `rename`(1), `copy`(1), `split`(1), `combine`(1), `concatenate`(1) — **13개 전 action 포괄** |
| `mcp__obsidian_vault__edit` | 18 | 12 | 2 | `patch`(8: `frontmatter`·`heading`·`block`), `window`(3: 치환·삭제), `append`(3), `at_line`(3: 삽입·교체), `from_buffer`(1) — **5개 전 action 포괄** |
| `mcp__obsidian_vault__graph` | 18 | 12 | 2 | `statistics`(3), `neighbors`(2), `backlinks`(2), `forwardlinks`(2), `path`(2), `traverse`(2), `search-traverse`(1), `advanced-traverse`(1), `tag-traverse`(1), `tag-analysis`(1), `shared-tags`(1) — **11개 전 action 포괄** |
| `mcp__obsidian_vault__dataview` | 18 | 12 | 2 | `query`(7: `TABLE`·`TASK`·`CALENDAR`·`LIST`), `list`(3), `metadata`(3), `validate`(3), `status`(2) — **5개 전 action 포괄** |
| `mcp__obsidian_vault__bases` | 18 | 12 | 2 | `export`(4: `json`·`csv`·`markdown`), `read`(3), `create`(3), `query`(3), `view`(3), `list`(2) — **스키마 6개 전 action 포괄 (`evaluate` 배제)** |
| `mcp__swallow_trader__swallow_get_stock_price` | 18 | 12 | 2 | `null` (18) — 현재가·등락폭·등락률·장운영 상태·단일 종목 보유 현황 |
| `mcp__swallow_trader__swallow_get_price_history` | 18 | 12 | 2 | `null` (18) — 7·10·15·20·30·50·75·100·120·200·250·365일 및 `toss`·`naver`·`all` 교차검증 |
| `mcp__swallow_trader__swallow_get_portfolio` | 18 | 12 | 2 | `null` (18) — 계좌 전체 총자산·예수금 잔고·보유 종목 목록·평균단가·실시간 평가손익 |
| `mcp__swallow_trader__swallow_order` | 18 | 12 | 2 | `null` (18) — `ticker`(6자리), `companyName`, `price`(>0), `tradeType`(`BUY`/`SELL`) 4개 필수 인자 전수 명시 |
| `mcp__swallow_trader__swallow_backtest` | 18 | 12 | 2 | `null` (18) — 60·120·180·250·360·400·500·750·900·1000거래일, RSI 매수/매도 기준, 익절/손절, 초기자본금 변주 |
| **합계** | **180** | **121** | **2** | Obsidian 35개 유효 enum 전수 커버 / Swallow `null` 90건 |

---

## 3. 표현 유형 및 핵심 경계 대조(Hard Contrasts) 설계

### 3.1 요청 표현 분포
- **직접 요청 (`direct`)**: `94`건 (명시적 파일 경로·쿼리·종목·파라미터 지정)
- **일상적·암시적 표현 (`everyday_implicit`)**: `56`건 (구어체 및 실무 맥락에서 도구 이름을 직접 부르지 않고 필요한 작업만 기술)
- **짧은 맥락 해결 요청 (`context_resolved`)**: `30`건 (호출 이전 `user`/`assistant` 대화에서 대상 파일·쿼리·종목·주문 단가가 모두 확정된 뒤 짧은 지시로 실행을 요청하며, `assistant` 메시지가 정답 도구 이름을 미리 발설하지 않음)

### 3.2 핵심 도구 간 경계 대조 (Hard Contrasts)
1. **`vault` vs `edit` vs `graph`**:
   - 파일 전체 덮어쓰기(`vault.update`), 파일 삭제(`vault.delete`), 파일 이름 변경(`vault.rename`), 파일 간 병합(`vault.combine`/`concatenate`)은 `vault`로 라우팅하고, 본문 내 특정 구문 치환·삭제(`edit.window`), 파일 끝 한 줄 추가(`edit.append`), 특정 줄 번호 편집(`edit.at_line`), 제목·블록·YAML frontmatter 부분 수정(`edit.patch`)은 `edit`로 구분했다.
   - 키워드 전문 검색(`vault.search`) 및 긴 파일 내 단락 발췌(`vault.fragments`)는 `vault`로, 노트 간 위키링크 1촌 이웃(`graph.neighbors`), 역방향 백링크(`graph.backlinks`), 다중 홉 탐색(`graph.traverse`/`search-traverse`), 두 노트 간 최단 링크 경로(`graph.path`) 및 공통 태그 그래프(`graph.shared-tags`)는 `graph`로 대비했다.
2. **`dataview` vs `bases` vs `vault`**:
   - `TABLE`/`LIST`/`TASK`/`CALENDAR` 등 DQL 쿼리 실행(`dataview.query`), DQL 구문 검증(`dataview.validate`), 단일 마크다운 노트의 Dataview 인덱스 메타데이터 추출(`dataview.metadata`), 플러그인 상태 확인(`dataview.status`)은 `dataview`로 지정했다.
   - `.base` 확장자 파일 목록(`bases.list`), `.base` YAML 설정 읽기(`bases.read`), 새 `.base` 생성(`bases.create`), `.base` 기반 속성 필터 쿼리(`bases.query`), 정의된 테이블·카드 뷰 열람(`bases.view`), CSV/JSON/Markdown 내보내기(`bases.export`)는 `bases`로 분리했다.
3. **`swallow_get_stock_price` vs `swallow_get_price_history` vs `swallow_get_portfolio` vs `swallow_order` vs `swallow_backtest`**:
   - 기존 `contrast-source-v1`의 종목(기아, LG화학, 한국전력, 삼성SDI, HMM, POSCO홀딩스)과 겹치지 않도록 **NAVER(035420), 셀트리온(068270), KB금융(105560), 두산에너빌리티(034020), 신한지주(055550), 크래프톤(259960), 한화에어로스페이스(012450), 삼성바이오로직스(207940), 현대모비스(012330), LG에너지솔루션(373220), 카카오뱅크(323410), 하이브(352820)** 12개 종목을 사용했다.
   - **단일 종목 실시간 시세·등락·장상태·개별 보유 여부**(`swallow_get_stock_price`) vs **과거 N일간 일자별 종가 시계열 및 토스·네이버 소스 교차검증**(`swallow_get_price_history`) vs **종목 지정 없는 계좌 전체 총자산·예수금·전 종목 평단가·평가손익**(`swallow_get_portfolio`) vs **필수 4개 인자(`ticker`, `companyName`, `price`, `tradeType`)가 완비된 수동 매수·매도 주문**(`swallow_order`) vs **과거 30~1000거래일 RSI 전략 수익률·MDD·승률 시뮬레이션**(`swallow_backtest`)을 1:1 대조쌍으로 구성했다.

---

## 4. 누수·중복·개인정보·계약 검증 결과

기존 빌더(`scripts/build_ten_tool_contrast.py`)와 감사 헬퍼(`apps/review/frozen_eval_audit.py`)를 재사용해 다음 항목을 검증했다. 작성 과정에서 기존 validation/test 요청문은 열람하거나 참조하지 않았다.

- **빌더 호환성 및 무결성 검증 (`build_ten_tool_contrast.build`)**:
  - `apps/review/training-source-v3-agy.jsonl` 단독 투입 시: `augmented/train` 240행(기존 60 + 신규 180, 도구당 24행, 시나리오 그룹 156개) 빌드 통과.
  - 기존 `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl`(180행)과 병합 투입 시: `augmented/train` 420행(기존 60 + v1 180 + v3-agy 180, 도구당 42행, 시나리오 그룹 216개) 빌드 통과 및 held-out `validation`/`test` 바이트 불변 확인.
- **정규화 입력 중복 및 분할 간 누수**: `0`건 (신규 180건 내부 중복 `0`건, 기존 train·contrast-v1 및 held-out 분할과의 정규화 서명/시나리오 ID 충돌 `0`건).
- **기존 학습 데이터 대비 최대 어휘·3-gram Jaccard 유사도**: `0.3871` (`0.55` 임계치 미만).
- **모델 입력 특징 경계 및 개인정보 위생**:
  - `request` 및 `context` 내 도구 ID(`mcp__*`) 또는 함수명(`swallow_*`) 노출: `0`건.
  - `context` 내 허용 역할(`user`, `assistant`) 외 키 또는 호출 후 결과(`tool_calls`, `outcome` 등) 포함: `0`건.
  - 자동화 프롬프트 패턴 및 비식별화되지 않은 개인정보(이메일, 토큰, 로컬 절대경로 등) 검출: `0`건.
- **검토 및 실행 상태 보존**: 전 문항 `review_status="needs_human_review"`, `execution_verified=false`, `source.kind="synthetic"`, `source.author="agy"` 유지.

---

## 5. 한계 및 후속 절차

- 본 데이터는 카탈로그 명세에 기반해 작성한 미검토 합성 학습 데이터(`needs_human_review`, `execution_verified=false`)이며, 실제 사용자 로그나 사람 검토가 완료된 정답이 아니다.
- `source.intended_contrast`와 `source.rationale`은 교차 검토(cross-review) 및 사람 검토를 돕기 위한 메타데이터이며, `build_ten_tool_contrast.py`에서 `state`(`request`, `context`)로 변환될 때 모델 입력 특징에서 완전히 제외된다.
- 통합 담당(Root / `dev-codex`)의 안내에 따라 타 작성자의 학습 보강 데이터에 대한 독립 교차 검토를 수행한 뒤, 승인된 사례만 최종 학습 데이터셋으로 병합한다.
