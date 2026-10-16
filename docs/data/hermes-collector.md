# Hermes 로그 수집기 v1

Python 3 표준 라이브러리만 사용한다. Hermes `state.db`를 `mode=ro`와 `query_only`로 열며, Hermes 설정·원본 DB를 변경하지 않는다. 수집기의 별도 SQLite에는 정제된 배치, 검토 후보, 보류 사유, 카탈로그와 체크포인트를 저장한다. 자동 학습·정답 확정·분할은 수행하지 않는다.

## 입력과 수집 단위

- 실제 Hermes 스키마: `sessions(id, parent_session_id, ended_at)`, `messages(id, session_id, role, content, tool_calls, tool_call_id)`. 추가 필드는 허용한다. `active`, `compacted`, `_compressed_summary`가 있으면 재작성된 기록을 보류한다.
- 종료된 세션의 assistant 메시지 하나가 선택 시점 하나다. 가장 가까운 이전 user 메시지가 `request`이며, 그 메시지를 제외한 선택 이전 메시지만 `context`에 넣는다. 현재 assistant의 설명, 현재 호출의 인자·결과와 이후 답변은 입력에 포함하지 않는다. 이전 단계의 결과는 이전 맥락으로 포함할 수 있다.
- 호출이 없는 assistant 응답도 수집하지만 `no_tool` 정답은 부여하지 않는다. 병렬 호출은 같은 사례의 `observed_choices`에 함께 저장한다.
- 부모를 따라 최상위 세션을 찾아 HMAC `session_group`을 만든다. 부모 누락·순환은 보류한다. 다른 DB에 있는 부모를 자동 연결하지 않는다.
- 결과는 동일 `tool_call_id`로 연결한다. JSON의 명시적인 `success`, `is_error`, `error`만 실행 관측값으로 해석한다. 일반 텍스트·결과 누락은 `unknown`, 성공과 오류가 섞이면 `mixed`다. 결과 텍스트와 현재 호출 인자는 저장하지 않는다. 시간 초과 같은 실행 실패를 도구 선택 오답으로 확정하지 않는다.

실제 스키마에서 선택 시점별 후보 스냅샷은 확인되지 않았다. 기본 실행은 `missing_historical_candidates`로 보류한다. `--reconstruct-candidates`를 명시하면 제공한 카탈로그 **전체**를 복원 후보로 사용하고 `candidate_tools_reconstructed: true`로 표시한다. 이는 당시 노출된 후보의 증거가 아니므로, 운영 정확도 평가용으로 바로 사용하지 않는다. 정확한 당시 후보를 자동 복원하는 어댑터는 아직 없다.

카탈로그는 기존 `tools.json` 형식(`catalog_version`, `tools`의 `id`, `server`, `name`, `input_schema`)으로 제공한다. 원본 MCP 캐시를 자동 가져오지 않는다. `mcp__server__name`과 Hermes의 `mcp_server_name`을 명시된 도구에만 연결하며, 모호한 alias를 거부한다. 알 수 없는 도구, MCP 외 도구, 잘못된 action, 필수 action 누락은 사례 전체를 보류한다. 사용한 카탈로그는 별도 상태 DB에 보관한다.

## 정제와 로컬 파일

이메일, URL, 문서·파일 경로, 전화번호, 주민등록번호 형태, 긴 숫자 사용자 ID, IP를 HMAC 가명으로 바꾼다. 알려진 API 토큰·Bearer·인증/비밀 키가 있는 줄과 PEM 개인 키를 제거한다. 원본 사용자·세션·메시지 ID와 호출 인자, 시스템 프롬프트는 출력하지 않는다. 호출 인자의 `action` enum만 검증해 보존한다.

이름·조직 내부 식별자·특수한 비밀 형식은 일반 정규식으로 모두 식별할 수 없다. 실제 사용 전 해당 식별자들을 로컬 `--redactions` JSON 문자열 목록으로 제공하고 정제 결과를 검토해야 한다. 예: `["테스트 사용자", "사내 고객 식별자"]`. 부분 문자열을 그대로 일치시켜 가명화한다. `sanitized: true`는 이 정책을 적용했다는 뜻이며 익명성 인증이 아니다. 미탐 가능성이 있는 자유 텍스트의 무인 배포는 검증하지 않았다.

상태 DB와 새 내보내기 파일은 권한 `0600`으로 생성한다. HMAC 키는 상태 DB에만 보관한다. 인증 토큰은 환경 변수로 받으며 DB·워크플로·Git에 저장하지 않는다. 원본 및 상태 파일, redactions와 내보내기는 저장소의 무시된 `data/local/` 같은 비공개 위치에 둔다. 기존 내보내기 파일을 덮어쓰지 않는다. 카탈로그는 정제된 신뢰 가능한 명세를 제공해야 한다.

## 실행

저장소 루트에서 경로를 실제 로컬 경로로 대체한다. 아래 `source.db`는 사용자가 지정한 실제 Hermes DB이며 수집기가 생성하지 않는다. `data/local/redactions.json`은 위 형식으로 준비한다.

```sh
python3 scripts/validate_dataset.py
python3 scripts/collect_hermes_logs.py once \
  --source /private/hermes/state.db \
  --state data/local/collector.db \
  --catalog datasets/hermes-routing-v1/tools.json \
  --redactions data/local/redactions.json \
  --reconstruct-candidates
```

한 번에 assistant 메시지를 최대 100개 검사하고 해당 배치를 저장·ACK한다. `--limit`은 1~1000이다. 기본 소비자는 `jery-dataset-v1`이다. `--reconstruct-candidates`를 빼면 후보 미확보 사례를 보류하는 안전한 경로를 확인할 수 있다. 보류 목록에는 가명 사례 ID와 고정 사유 코드만 남기며 원본 텍스트를 저장하지 않는다. `once`는 원문 대신 저장 건수와 보류 사유별 누적 건수만 출력한다.

같은 옵션에서 `once`를 `status`로 바꾸면 상태를 확인하고, `export --output data/local/review.jsonl`로 바꾸면 검토 후보를 JSONL로 내보낸다. 내보낸 각 행은 API 사례 필드와 `decision: null`, `acceptable_choices: []`, `review_status: needs_human_review`를 포함한다. 사람 검토 결과가 저장된 행은 그대로 내보낸다. 이 파일은 합성 v1 학습 JSONL과 다르다.

HTTP 서버:

```sh
# 환경의 비밀 관리 도구로 JERY_COLLECTOR_TOKEN을 16자 이상 ASCII 값으로 설정한다.
python3 scripts/collect_hermes_logs.py serve \
  --source /private/hermes/state.db \
  --state data/local/collector.db \
  --catalog datasets/hermes-routing-v1/tools.json \
  --redactions data/local/redactions.json \
  --reconstruct-candidates
```

기본 바인딩은 `127.0.0.1:8080`이다. 두 API 모두 `Authorization: Bearer <token>`이 필요하다. `--host`, `--port`로 바꿀 수 있다. stdlib 서버 자체는 TLS를 제공하지 않는다. n8n에서 접근 가능한 로컬 네트워크/역방향 프록시와 인증 credential을 설정한다. 실제 n8n 배포와 주기 활성화는 자동 수행하지 않는다. [API 계약](../contracts/log-candidates-api-v1.md), [n8n 설정](../../workflows/n8n/README.md)을 함께 따른다.

## 저장과 재시도

GET은 정제된 배치 스냅샷을 영속화하지만 체크포인트를 전진시키지 않는다. 같은 소비자의 ACK 전 GET은 동일 배치를 반환한다. 대기 중 `limit`을 줄여도 기존 배치는 그대로 반환하므로, n8n의 limit은 해당 배치 ACK 후 변경한다.

POST는 서버가 발행한 스냅샷과 전체 요청을 비교한다. 소비자·배치·커서·본문·참조·미검토 라벨이 다르면 거부한다. SQLite 쓰기 트랜잭션에서 후보 삽입, 검사한 원본의 가명 ID 기록, 체크포인트 갱신, 배치 완료 처리를 함께 수행한다. 부분 저장 실패는 전체 롤백한다. 동일 배치 ACK는 같은 성공 영수증을 반환한다. 오래된 ACK는 체크포인트를 되돌리지 않는다. 이미 있는 검토 후보는 덮어쓰지 않는다. 소비자별 체크포인트는 독립적이고 후보 ID는 공유한다.

빈 배치도 ACK해야 보류된 기록 이후로 진행한다. 열린 세션은 읽지 않으며 나중에 종료되면 메시지 ID가 작아도 발견한다. 보류 기록의 가명 ID와 사유는 상태 DB `quarantine` 테이블에 남는다. 같은 상태에서는 보류 사례를 매번 재처리하지 않는다. 소스 경로·카탈로그·정제 목록·복원 정책이 달라지면 기존 상태 재사용을 거부한다. 재검사는 새 상태 DB로 수행하며, 기존 사람 검토 결과의 자동 이전/병합은 아직 없다. 이전 상태와 검토 파일은 보존해야 한다.

## 검증과 한계

```sh
python3 -m unittest discover -s tests/data -v
python3 -m unittest discover -s tests/integration -v
python3 scripts/validate_dataset.py
```

테스트는 임시 합성 Hermes SQLite로 원본 바이트 보존, 정제, 부모 묶음, 호출 전 입력, 무호출·병렬 호출, 보류, 재시작·동시 GET/ACK, 실패 롤백, 본문 변조 거부, 검토 보존과 HTTP 인증을 검사한다. 통합 테스트에는 Node.js가 필요하며 기존 n8n 검증 JavaScript를 실제 실행해 HTTP 왕복을 확인한다. 외부 Python 의존성은 없다.

현재 매 GET마다 종료된 메시지 ID를 다시 훑고 필요한 세션을 메모리에 읽는다. 대용량에서는 로컬 원본 인덱스와 세션 단위 스트리밍이 후속 작업이다. 사례는 256 KiB를 넘으면 `oversized_record`로 보류하며 배치는 약 8 MiB로 제한한다(HTTP POST 상한 16 MiB). 실제 후보 스냅샷 복원, 실행 결과 원문 검토 화면, 검토 결과 병합, 학습용 내보내기, 다중 프로필 간 그룹 연결은 포함하지 않는다.
