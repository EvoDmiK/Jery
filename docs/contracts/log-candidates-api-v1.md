# 로그 검토 후보 API v1 — 구현 예정 계약

이 문서는 `workflows/n8n/hermes-logs-to-review.json`이 요구하는 로컬 수집기 계약이다. 현재 해당 API 구현은 없다. 기존 합성 학습 데이터의 스키마를 변경하지 않는다.

## GET /v1/routing-candidates

Query: `consumer=jery-dataset-v1`, `limit=100`. Header Auth를 사용한다. 응답은 JSON 본문이며 다음 구조다. 아래 사례는 API 형식 설명용으로 만든 예시이고 실제 로그에서 추출한 데이터가 아니다.

```json
{
  "schema_version": "jery-log-candidates-v1",
  "consumer": "jery-dataset-v1",
  "batch_id": "batch_001",
  "next_cursor": "cursor_001",
  "sanitized": true,
  "examples": [
    {
      "id": "example_001",
      "session_group": "session_001",
      "request": "볼트에서 Jery 관련 노트를 찾아줘",
      "context": [],
      "candidate_tools": ["mcp__obsidian_vault__vault"],
      "tool_catalog_version": "example-catalog-v1",
      "candidate_tools_reconstructed": true,
      "observed_choices": [
        {"tool": "mcp__obsidian_vault__vault", "action": "search"}
      ],
      "outcome": "unknown",
      "source": "hermes_log"
    }
  ]
}
```

- `context`에는 선택 이전의 정제된 메시지만 포함한다. 각 항목은 `role` (`user`, `assistant`, `tool`)과 문자열 `content`를 가진다. 현재 호출의 결과나 이후 답변은 입력 맥락에 넣지 않는다.
- `observed_choices`는 관측된 호출 목록이다. 병렬 호출은 여러 항목, 무호출 사례는 빈 배열이다. 검토된 정답이 아니다.
- `outcome`은 `unknown`, `ok`, `error`, `mixed` 중 하나다. 선택의 적절성과는 별개의 실행 관측값이다.
- `session_group`은 원본 세션과 파생 세션을 함께 묶는 가명 식별자다.
- `candidate_tools_reconstructed`는 당시 후보를 원본 기록에서 확보했는지, 나중에 복원했는지를 구분한다.
- 수집기는 해당 버전의 명세를 보관하고 도구 ID 및 action을 검증한다. 당시 후보를 신뢰성 있게 확보하지 못한 사례는 별도 보류 대상으로 관리한다.
- `batch_id`와 사례 ID는 재시도에도 동일해야 한다. ACK 전까지 동일 소비자의 미완료 배치를 다시 반환하고, GET 자체로 커서를 전진시키지 않는다.
- 수집기는 개인정보와 비밀을 제거한 뒤에만 `sanitized: true`를 반환한다. n8n 실행 기록에도 정제된 데이터만 들어가야 한다.

## POST /v1/review-batches

GET 배치에서 `sanitized`를 제외하고, 각 사례에 다음 필드를 부여한 JSON을 받는다.

```json
{
  "decision": null,
  "acceptable_choices": [],
  "review_status": "needs_human_review"
}
```

실제 요청은 위 필드만이 아니라 `schema_version`, `consumer`, `batch_id`, `next_cursor`와 전체 `examples` 배열을 포함한다. 워크플로가 whitelist로 재구성하므로 원본의 임의 추가 필드는 저장 요청에 들어가지 않는다.

서버는 소비자와 배치의 일치를 검증하고, 한 트랜잭션에서 다음을 수행한다.

1. 배치 ID와 사례 ID로 검토 후보를 중복 없이 저장한다.
2. 기존 사람 검토 결과가 있으면 보존한다. 재수집이 검토 상태와 정답을 초기화하지 않는다.
3. 배치의 모든 사례를 영속 저장한 뒤 소비자 체크포인트를 갱신한다.
4. 빈 배치도 처리해 필터링된 원본 기록 이후로 진행할 수 있게 한다.

동일 배치의 재시도와 동시 호출은 동일한 성공 응답을 반환한다. 오래된 배치가 체크포인트를 되돌리면 안 된다. 실패 시 커서를 전진시키지 않고 HTTP 오류를 반환한다.

성공 응답:

```json
{
  "batch_id": "batch_001",
  "next_cursor": "cursor_001",
  "committed": true
}
```

이 단계에서는 최종 정답, 분할과 manifest를 생성하지 않는다. 검토 완료 이후 별도 내보내기 단계가 학습 데이터 스키마로 변환한다.
