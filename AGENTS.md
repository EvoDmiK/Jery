# Jery 에이전트 작업 규칙

## 프로젝트 목표와 현재 상태

Jery는 사용자 요청, 이전 대화, 후보 도구 명세를 받아 다음 MCP 도구를 선택하는 경량 로컬 모델 프로젝트다. 결정 유형은 `select_tools`, `no_tool`, `clarify`, `unsupported`다.

현재 구현은 한국어 합성 데이터셋 130개와 재생성·검증 스크립트다. 읽기 전용 Hermes SQLite 수집·정제, 검토 후보 저장과 체크포인트 API도 구현되어 있다. 모델, 추론 서버와 검토 UI는 아직 없다. 수집기 실행과 한계는 `docs/data/hermes-collector.md`를 따른다. 기능 완료 여부는 코드와 검증 결과로 확인하고, 계획을 구현된 기능처럼 보고하지 않는다.

## 역할과 파일 소유권

역할은 폴더 표시 이름 대신 `git branch --show-current`로 확인한 브랜치에 따라 결정한다. 현재 자식 워크트리의 표시 이름은 모두 `main`이지만 브랜치는 서로 다르다.

| 브랜치 | 담당 | 기본 수정 범위 |
|---|---|---|
| `dev-codex` | Codex: 통합, 데이터 수집과 공통 계약 | `datasets/`, `scripts/`, `src/jery/data/`, `src/jery/runtime/`, `tests/data/`, `tests/integration/`, `docs/data/`, `docs/contracts/`, 루트 공통 설정과 문서 |
| `dev-claude` | Claude: 기준 모델, 학습과 평가 | `src/jery/model/`, `src/jery/evaluation/`, `tests/model/`, `tests/evaluation/`, `configs/model/`, `docs/model/`, `requirements-model.txt` |
| `dev-agy` | Antigravity: 데이터 검토 도구와 결과 시각화 | `apps/review/`, `tests/review/`, `docs/review/`, `requirements-review.txt` |

위 경로 중 아직 없는 디렉터리는 담당 기능 구현 시 생성한다. 역할 표 자체는 실행 명령이 아니며, 구체적인 작업 지시가 주어지면 해당 범위에서 수행한다. 범위 밖 변경이 필요하면 통합 담당에게 경로, 이유와 필요한 인터페이스를 전달하고, 그동안 독립적으로 진행할 수 있는 작업을 계속한다.

다른 브랜치에서는 먼저 사용자의 작업 지시를 확인한다. 임의로 다른 에이전트 역할을 맡거나 공유 파일을 수정하지 않는다. 사용자의 명시적인 지시는 이 문서의 역할 분담보다 우선한다.

## dev-codex: 데이터와 통합

- 데이터 스키마, 도구 카탈로그 버전, 모델·UI 입출력 계약을 관리한다. 합의된 계약은 `docs/contracts/`에 기록한다.
- Hermes 로그를 읽기 전용으로 수집해 요청, 호출과 결과를 연결한다. 운영 DB나 Hermes 설정을 직접 변경하지 않는다.
- 원본 로그에서 정제·가명화된 검토 대상을 만들고, 검토 결과를 학습 데이터로 반영한다.
- 시나리오·세션 분할, 중복 제거, 데이터 검증과 출처 추적을 담당한다.
- 공통 의존성, 패키징, 루트 `README.md`, `AGENTS.md`와 CI를 관리한다.
- 자식 브랜치 결과를 검토하고 통합 테스트를 실행한 뒤 통합한다.

첫 구현 과제의 예: 실제 Hermes 로그를 정제된 `needs_review` 사례로 변환하는 수집기. 후보 목록을 복원했으면 복원 여부를 표시하고, 기존 선택을 검토된 정답으로 승격하지 않는다.

## dev-claude: 모델과 평가

- 기존 JSONL과 `tools.json`을 읽는 기준 모델을 구현한다. 기존 데이터를 수정해 모델에 맞추지 않는다.
- 먼저 간단한 로컬 기준 모델로 비교 기준을 확보하고, 더 큰 모델은 효과와 실행 비용을 비교해 도입한다.
- 모델 입력에는 요청, 이전 맥락과 후보 명세만 사용한다. 정답, 검토 상태, 출처, 분할 이름과 정답을 암시하는 ID는 특징으로 사용하지 않는다.
- 모든 결정 유형을 지원하고, 선택한 도구가 해당 사례의 후보에 포함되는지 확인한다.
- 학습·전처리 통계는 학습 분할에서만 계산한다. 하이퍼파라미터는 검증 분할로 정하고, 평가 분할로 반복 튜닝하지 않는다.
- 결정 유형별 precision/recall, 도구 선택 정확도, 불필요한 호출 비율과 로컬 추론 시간을 보고한다. 상위 후보를 반환하는 모델이면 recall@k도 보고한다.
- 데이터 버전, 모델 설정, 난수 시드, 실행 명령과 실행 환경을 기록한다.
- 현재 합성 데이터의 미검토 정답으로 수행한 실험은 탐색 실험으로 표시한다. 실제 사용 정확도나 검토된 정답 성능으로 주장하지 않는다.

첫 구현 과제의 예: 한 명령으로 학습·평가할 수 있는 기준 모델과, 통합 담당이 호출할 수 있는 예측 함수. 외부 서비스 없이 작은 데이터에서 실행되어야 한다.

## dev-agy: 검토 도구와 시각화

- 정제된 JSONL, 도구 카탈로그와 모델 평가 결과를 읽는 로컬 검토 도구를 구현한다.
- 요청, 호출 전 맥락, 후보 명세, 관측된 선택과 정답 후보를 사람이 함께 확인할 수 있게 한다.
- 검토자는 결정 유형, 허용 도구와 action을 수정하고 검토 의견을 남길 수 있어야 한다.
- 원본 데이터셋을 덮어쓰지 않고 별도의 검토 결과 파일을 내보낸다. 통합 담당이 정답을 병합한다.
- 미검토·검토 완료 건수, 결정 유형별 분포와 오류 사례를 표시한다. 지표는 실제 입력 파일에서 계산한다.
- 검토 상태와 실행 검증 상태는 별개로 표시한다. 검토 완료를 실제 실행 성공으로 바꾸지 않는다.
- UI 렌더링 시 로그 텍스트는 데이터로 취급한다. HTML·스크립트나 도구 실행 명령으로 해석하지 않는다.
- 입력 파일과 출력을 로컬에 유지한다. 원본 DB를 직접 열거나 서버에 업로드하는 기능은 기본 범위에 포함하지 않는다.

첫 구현 과제의 예: 기존 합성 사례를 읽고 정답을 검토한 뒤 사례 ID 기반의 수정 결과를 내보내는 최소 도구. 모델이 준비되기 전에도 사용할 수 있어야 한다.

## 공통 데이터 계약

- 현재 필드와 의미는 `datasets/hermes-routing-v1/README.md`, 카탈로그와 manifest를 기준으로 한다.
- 도구 ID는 `mcp__{server}__{name}`이다. 도구 명세는 사례의 `tool_catalog_version`에 연결한다.
- `acceptable_choices`는 허용되는 선택의 대안 목록이다. 여러 단계의 실행 계획으로 해석하지 않는다.
- `unsupported`는 해당 사례의 후보 도구로 처리 불가라는 뜻이다.
- schema나 라벨 의미가 바뀌면 버전을 올리고 마이그레이션 방법을 기록한다. 기존 v1 파일을 조용히 재해석하지 않는다.
- 검토 도구의 출력에는 사례 ID, 카탈로그 버전, 수정된 결정·허용 선택, 검토 상태와 의견을 포함한다. 정확한 필드와 상태 값은 공통 계약에 맞춘다.
- 모델과 검토 도구 간 직접 연결은 공통 계약이 정해진 뒤 구현한다. 그전에는 담당 범위의 함수나 파일 어댑터로 독립 작업한다.

## 실제 로그 수집과 학습 원칙

- 모델 입력에는 도구 선택 이전 정보만 넣는다. 해당 호출의 결과와 이후 답변은 검토 근거로만 사용한다.
- 관측된 호출과 검토된 정답을 별도 필드로 보존한다. 성공 호출도 검토 없이 정답으로 확정하지 않는다.
- 타임아웃·서버 장애와 잘못된 선택을 구분한다. 실패 사례를 일괄 삭제하지 않는다.
- 같은 대화, 파생 세션과 표현 변형을 분할 간에 섞지 않는다.
- 호출이 없던 대화도 검토해 `no_tool`·`clarify` 사례를 확보한다.
- 인증 토큰, 쿠키, 실제 이메일과 개인 식별자는 학습 파일·출력·커밋에 포함하지 않는다. 참조가 필요한 문서 경로와 사용자 ID는 일관되게 가명화한다.
- 원본 로그, DB 사본, 가중치와 대용량 실험 결과는 Git에 추가하지 않는다. 저장 위치와 접근 방식은 통합 담당과 정한다.

## 브랜치와 작업 절차

1. 작업 시작 시 현재 브랜치와 작업 트리 변경사항을 확인한다. 다른 사람의 변경을 삭제하거나 덮어쓰지 않는다.
2. 각 에이전트는 자기 워크트리와 담당 브랜치에서 작업한다. 부모나 다른 자식 워크트리의 제품 파일을 직접 수정하지 않는다.
3. 공통 변경이 필요하면 통합 담당에게 요청한다. 담당별 의존성은 자기 범위의 파일에 기록한다.
4. 필요한 upstream 변경은 통합 담당이 제공한 기준 커밋을 확인해 반영한다. 진행 중 작업을 강제로 reset하거나 rebase하지 않는다.
5. 작업 지시에 커밋·푸시가 포함되어 있으면 자기 브랜치에 수행한다. 통합 브랜치나 `main`으로 직접 머지하거나 강제 푸시하지 않는다.
6. 작업 중단·완료 시 무엇을 구현했고 무엇이 남았는지 기록한다. 워크트리나 브랜치를 임의로 삭제하지 않는다.

AGENTS.md 작성이나 역할 확인만으로 다른 에이전트에게 작업을 보내거나 새 에이전트를 실행하지 않는다. 실제 작업 배정은 사용자의 지시 범위에 따른다.

## 검증과 완료 보고

데이터를 소비하거나 변경하는 작업은 기존 데이터 검증부터 확인한다.

```sh
python3 scripts/validate_dataset.py
```

`build_seed_dataset.py`는 기존 분할과 manifest를 덮어쓰므로 합성 데이터를 재생성하는 작업에만 실행한다. 모델과 검토 도구는 담당 기능에 맞는 테스트를 실행한다. 설명 문서만 변경한 경우에는 문서 내용, 링크와 `git diff --check`를 확인한다.

모델 검증에는 후보 밖 도구 선택 방지와 입력에 정답 정보가 섞이지 않는지 확인하는 사례를 포함한다. 검토 도구는 내보낸 결과를 다시 읽을 수 있는지, 원본이 보존되는지와 잘못된 도구·action 참조를 거부하는지 확인한다.

완료 보고에는 다음을 포함한다.

- 변경 요약과 담당 경로
- 커밋 SHA와 브랜치, 푸시 여부
- 실행·재현 명령과 실제 검증 결과
- 데이터·계약 버전과 필요한 의존성
- 남은 한계, 통합 담당이 해야 할 작업

## 문서 배포

이 파일은 `dev-codex`에서 관리하는 공통 규칙이다. 자식 워크트리에 복사된 파일은 동일한 안내용 사본이며 독립적으로 수정하지 않는다. 통합 담당이 공통 파일을 커밋해 제공하면 사본과 충돌하지 않도록 확인한 뒤 반영한다.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Ponytail — 최소 구현 원칙

코드를 작성하기 전에 문제와 호출 흐름을 확인하고, 다음 순서에서 해결되는 첫 방법을 선택한다.

1. 실제로 구현이 필요한지 확인한다.
2. 저장소의 기존 함수와 패턴을 재사용한다.
3. 표준 라이브러리를 사용한다.
4. 플랫폼 기본 기능을 사용한다.
5. 이미 설치된 의존성을 사용한다.
6. 간단한 코드로 해결한다.
7. 남는 부분만 최소한으로 구현한다.

요청하지 않은 추상화, 의존성, boilerplate를 추가하지 않는다. 버그는 관련 호출 경로를 확인해 공통 원인을 수정한다. 입력 검증, 데이터 보존과 사용자가 요청한 기능을 생략하지 않는다. 비단순 로직은 실제 동작을 확인하는 최소 검증을 남긴다.

상세 규칙은 `.agents/rules/ponytail.md`에 있다. 이 원칙은 위의 역할 분담과 공통 계약에 함께 적용한다.
