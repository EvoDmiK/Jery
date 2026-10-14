# Hermes 로그 수집 워크플로

파일: [hermes-logs-to-review.json](hermes-logs-to-review.json)

실제 Hermes 로그에서 정제된 검토 후보를 주기적으로 수집하는 **n8n 가져오기용 초안**이다. 수집기 API는 아직 구현되지 않았으며, 이 파일만으로 데이터가 생성되지 않는다. n8n에 배포하거나 활성화하지 않았다.

```text
Manual Trigger / Every 5 Minutes
  → Configuration
  → Fetch Sanitized Log Batch
  → Validate Review Candidates
  → Persist Review Batch and Checkpoint
  → Check Commit Receipt
```

## 설정

1. n8n의 파일 가져오기 기능으로 JSON을 가져온다.
2. `Configuration`의 `collector_url`을 n8n에서 접근 가능한 수집기 주소로 바꾼다. 기본 `http://jery-collector:8080`은 예시이며 실제 서비스가 아니다. Docker의 `localhost`는 n8n 컨테이너 자체를 가리킨다.
3. 두 HTTP Request 노드에 수집기용 Header Auth credential을 선택한다. 예를 들어 헤더 이름 `Authorization`, 값 `Bearer <token>`을 credential에 저장한다. JSON에는 인증 값이 없다.
4. [수집기 API 계약](../../docs/contracts/log-candidates-api-v1.md)을 구현한다.
5. `Manual Trigger`로 수집·검증·저장 응답을 확인한 뒤 주기 실행을 활성화한다. 시간대는 `Asia/Seoul`이다.

HTTP 단계는 60초 timeout과 최대 3회 재시도를 사용한다. 잘못된 배치는 전체 실행을 실패시키며 저장과 체크포인트 갱신 단계로 넘어가지 않는다. 저장 API가 멱등성을 보장해야 재시도로 중복이 생기지 않는다.

## 역할과 한계

- Python 수집기: DB 읽기 전용 접근, 호출 전 맥락 복원, 개인정보 제거, 도구 카탈로그 보관과 참조 검증.
- n8n: 주기 실행, 정제된 배치 가져오기, 형식 검증과 저장 요청.
- 검토 도구: 사람이 `decision`과 `acceptable_choices`를 검토해 부여.
- 별도 내보내기 단계: 검토 결과를 세션별로 분할하고 검증해 학습 JSONL로 생성.

실행 결과가 성공이어도 정답으로 확정하지 않는다. 이 워크플로는 모든 사례를 `needs_human_review`, `decision: null`, `acceptable_choices: []`로 저장한다. 미검토 후보 형식은 기존 v1 학습 데이터셋과 다르므로 바로 학습에 넣지 않는다.

코드 노드는 출처·후보 참조·최소 필드를 검증하고 허용된 필드만 저장 요청에 포함한다. `sanitized: true`는 수집기의 선언이며 자동 개인정보 탐지나 제거 기능이 아니다. 데이터는 n8n에 도달하기 전에 정제되어야 한다.

JSON 파싱, 연결 참조, JavaScript 검증 분기와 저장 응답 확인 로직을 로컬에서 확인했다. 실제 n8n 가져오기·실행과 API 연결은 아직 검증하지 않았다.
