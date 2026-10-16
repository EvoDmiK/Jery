# Jery

**MCP 도구 선택을 위한 경량 로컬 학습 결정 모델 프로젝트.**

Jery는 사용자 요청, 이전 대화, 사용 가능한 도구 명세를 입력으로 받아 다음 단계에 사용할 도구를 선택하는 것을 목표로 한다. 도구를 호출할 필요가 없거나 추가 정보가 필요한 상황, 현재 도구로 처리할 수 없는 요청도 구분한다.

## 현재 상태

현재 저장소에는 초기 도구 10개의 합성 데이터 v1과 전체 MCP 도구 96개의 합성 데이터 v2, 재생성·검증 스크립트가 있다. Hermes SQLite를 읽기 전용으로 수집·정제하고 검토 후보와 체크포인트를 저장하는 로컬 API도 구현되어 있다. 모델 학습 코드, 학습된 가중치와 추론 서버는 아직 없으며, 실제 n8n 배포는 수행하지 않았다.

## 결정 유형

| 결정 | 의미 | 요청 예시 |
|---|---|---|
| `select_tools` | 후보에서 적합한 다음 도구 선택 | “볼트에서 MCP 관련 노트를 찾아줘.” |
| `no_tool` | 도구 호출 없이 답변 가능 | “이 문장을 영어로 번역해줘: 나는 데이터를 모으고 있다.” |
| `clarify` | 대상이나 필요한 정보 확인 | “그 종목 현재가 조회해줘.” |
| `unsupported` | 현재 후보 도구로 처리 불가 | “내 Google 캘린더에 회의를 등록해줘.” |

판단 기준은 해당 사례의 후보 도구 목록이다. 전체 Hermes에 존재하는 기능도 후보 목록에 없으면 `unsupported`가 될 수 있다. 도구 선택은 현재 단계의 판단이며, 전체 작업 계획이나 호출 인자 생성은 초기 데이터셋의 범위에 포함하지 않는다.

## 전체 MCP 분류 데이터셋

[전체 MCP 분류 v2](datasets/mcp-routing-v2/README.md)는 Hermes `hybrid-v2` 등록 설정과 도구 명세를 기준으로 MCP 서버 5개·도구 96개를 포함한다. Grafana 62개, Swallow 13개, Obsidian 10개(리소스 유틸리티 포함), 한국 법률 10개, Kestrel 1개다. 실제 대화 로그 없이 만든 합성 사례 1,162개이며 학습 668개·검증 248개·평가 246개로 나눈다.

전체/일부 후보, 지원 도구가 빠진 경우와 맥락에 따라 판단이 바뀌는 경우를 포함한다. 서버 라벨은 도구 ID에서 도출한다. 전부 미검토 정답이며 실제 연결·실행 성공이나 운영 성능을 뜻하지 않는다. 표현 변형을 함께 묶으므로 검증·평가의 도구 커버리지는 각각 85개·84개다. 기존 v1은 보존하며 v1과 v2를 단순 합치지 않는다.

```sh
python3 scripts/build_mcp_dataset.py
python3 scripts/validate_mcp_dataset.py
```

## 초기 데이터셋

[Hermes routing v1](datasets/hermes-routing-v1/README.md)은 Hermes `hybrid-v2`의 로컬 MCP 스키마 캐시에서 추출한 도구 명세를 바탕으로 작성한 한국어 합성 데이터셋이다.

| 항목 | 구성 |
|---|---|
| 도구 | 10개: Obsidian 4개, Swallow 5개, Kestrel 1개 |
| 사례 | 130개: 원본 시나리오 65개 × 표현 변형 2개 |
| 결정 유형 | 도구 선택 100개, 나머지 유형 각 10개 |
| 분할 | 학습 78개 / 검증 26개 / 평가 26개 |
| 검토 상태 | 모든 사례가 `needs_human_review` |
| 실행 검증 | 모든 사례가 `execution_verified: false` |

같은 시나리오의 표현 변형은 동일한 분할에 들어간다. 모든 사례는 같은 10개 도구를 후보로 사용하며, 후보 목록의 순서는 사례별로 섞는다.

### 포함된 도구

| MCP 서버 | 도구 이름 |
|---|---|
| `obsidian_vault` | `vault`, `edit`, `view`, `graph` |
| `swallow_trader` | `swallow_get_stock_price`, `swallow_get_stock_news`, `swallow_get_price_history`, `swallow_get_portfolio`, `swallow_get_trades` |
| `kestrel` | `get_codex_reset_status` |

도구 ID는 `mcp__{server}__{name}` 형식이다. 실제 설명과 입력 스키마는 [tools.json](datasets/hermes-routing-v1/tools.json)에 저장되어 있다. Obsidian 도구의 `action`은 작업 종류를 구분하는 보조 라벨이다.

### 사례 구조

각 JSONL 파일은 한 줄에 하나의 JSON 객체를 저장한다.

| 필드 | 역할 |
|---|---|
| `id`, `scenario_id`, `split` | 사례 식별자, 원본 시나리오와 분할 |
| `request`, `context` | 사용자 요청과 이전 대화. v1의 이전 대화는 빈 배열이다. |
| `candidate_tools` | 선택 가능한 도구 ID 목록 |
| `tool_catalog_version` | 연결할 도구 명세의 버전 |
| `decision` | 네 가지 결정 유형 중 하나 |
| `acceptable_choices` | 허용되는 다음 선택의 목록. 선택마다 `tool`과 선택적 `action`을 포함한다. |
| `source`, `review_status`, `execution_verified` | 데이터 출처, 정답 검토와 실행 검증 상태 |

`acceptable_choices`의 항목은 허용되는 대안이며 순차 실행 계획이 아니다. `select_tools` 외의 결정에서는 빈 배열이다.

## 사용 방법

Python 3가 필요하며 외부 패키지 설치 없이 표준 라이브러리만으로 실행한다. 저장소 루트에서 다음 명령을 사용한다.

기존 데이터셋 검증:

```sh
python3 scripts/validate_dataset.py
```

정상 출력:

```text
PASS: 130 examples, 10 tools, 65 isolated scenarios
```

초기 합성 데이터 재생성 및 검증:

```sh
python3 scripts/build_seed_dataset.py
python3 scripts/validate_dataset.py
```

빌더는 저장된 `tools.json`과 코드에 정의된 시나리오를 사용해 분할 파일과 `manifest.json`을 덮어쓴다. 같은 입력에서는 사례와 분할이 동일하게 생성된다. Hermes 실행이나 MCP 서버 연결은 필요하지 않다.

검증 스크립트는 사례·요청 중복, 시나리오의 분할 간 누출, 도구 참조, action enum, 분할별 도구 커버리지, 건수와 SHA-256 체크섬을 확인한다. 정답의 의미적 정확성은 별도 검토가 필요하다.

## 저장소 구조

```text
Jery/
├── README.md
├── datasets/
│   └── hermes-routing-v1/
│       ├── README.md
│       ├── tools.json
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       └── manifest.json
└── scripts/
    ├── build_seed_dataset.py
    └── validate_dataset.py
```

## 에이전트 개발 도구

Codex, Claude Code, Antigravity에서 사용할 프로젝트별 Graphify 스킬과 설정을 포함한다. 역할과 수정 범위는 [AGENTS.md](AGENTS.md)를 따른다.

- **Graphify**: 코드 구조와 관계를 `graphify-out/`에 저장하고 `query`, `path`, `explain`으로 탐색한다. 현재 그래프는 로컬 구조 추출로 생성했으며, 문서 의미 추출은 수행하지 않았다.
- **Ponytail**: 기존 코드·표준 라이브러리·플랫폼 기능을 먼저 활용하고 필요한 부분만 구현하는 개발 규칙이다. 공통 규칙은 `AGENTS.md`, 상세 규칙은 `.agents/rules/ponytail.md`와 `.claude/rules/ponytail.md`에 있다.

Graphify CLI가 설치된 환경에서는 다음 명령으로 프로젝트 설정을 적용할 수 있다.

```sh
graphify install --project --platform codex
graphify install --project --platform claude
graphify install --project --platform antigravity
```

로컬 구조 그래프 생성과 조회:

```sh
graphify update . --no-cluster
graphify cluster-only . --no-label --no-viz
graphify export wiki
graphify query "build_seed_dataset validate_dataset"
```

생성물은 `.gitignore`에서 제외한다. 각 워크트리에서 다시 생성하고, `graphify-out/wiki/index.md`로 탐색할 수 있다.

## 실제 사용 로그로 확장하기

[Hermes 수집기](docs/data/hermes-collector.md)는 종료된 세션을 읽고 정제된 미검토 후보를 저장한다. 실행 방법, API, 후보 복원과 정제의 한계는 해당 문서를 따른다. 기존 합성 데이터는 변경하지 않는다. 후속 학습 데이터는 다음 기준을 따른다.

1. Hermes 대화 기록에서 요청, 도구 호출과 결과를 연결한다.
2. 각 도구 선택 시점을 하나의 사례로 만들고, 입력에는 호출 이전의 정보만 포함한다.
3. 기존 에이전트의 선택을 관측값으로 저장하고, 검토 후 허용되는 정답을 부여한다.
4. 호출 당시 후보 도구와 명세 버전을 기록한다. 복원한 후보 목록은 별도로 표시한다.
5. 개인정보와 인증 정보를 제거하고, 같은 세션과 파생 대화를 동일한 분할에 묶는다.

정상 실행 여부와 선택의 적절성은 별도로 판단한다. 타임아웃 같은 실행 오류만으로 잘못된 도구 선택이라고 분류하지 않는다. 도구를 사용하지 않은 대화도 검토해 `no_tool`과 `clarify` 사례를 확보한다.

## 한계와 다음 단계

현재 데이터는 수집 구조와 초기 실험을 위한 작은 합성 데이터셋이다. 실제 사용 로그를 포함하지 않으며, 정답은 사람 검토 전의 초안이다. 여러 도구 조합, 대화 맥락에 따른 선택, 실행 실패 이후의 판단도 포함하지 않는다. 반복되는 작업 유형과 표현 때문에 현재 평가 분할만으로 운영 성능을 추정하기 어렵다.

다음 단계는 정답 검토, 실제 사용 사례 수집, 별도 실제 요청 평가 세트 구성, 기준 모델의 학습·평가와 로컬 추론 구현이다.
