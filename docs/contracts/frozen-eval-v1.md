# 고정 평가셋 v1 계약

현재 모델의 과제는 고정 10개 후보 중 단일 도구 선택이다. 목표는 기존 학습·검증·평가와 분리된 실제 요청 약 100개(도구별 10개)이며, 확보·정답 검토·동결·모델 평가를 별도 상태로 기록한다. 목표 수를 확보하지 못하면 부족분을 그대로 보고하고 합성 자료를 실제 자료로 대체하지 않는다.

## 수집 후보와 동결 파일

`collect_frozen_eval_candidates.py`의 private `candidate-manifest.json`은 `frozen-eval-candidates-v1`이며 동결 manifest가 아니다. 이것은 종료된 대화의 실제 도구 실행을 재현하는 기존 collector와 다르다. 새 수집 경로는 root Discord user 메시지 중 active=1·compacted=0인 문장만 읽고 원본 DB는 query_only 모드로 연다. source/session ID는 HMAC으로 가명화한다. 열린 세션의 요청도 독립 문장 후보로만 수집하며 context는 비워 둔다. 과거 tool_calls·결과·답변은 입력이나 정답 근거에 쓰지 않는다. 문맥 의존 요청은 검토 후 제외하거나 별도 전처리 계약으로 확보해야 한다.

`frozen-eval-v1` 폴더 파일:

- `cases.jsonl`: id, scenario_id, split=evaluation, request, context, candidate_tools(고정10), tool_catalog_version, decision, acceptable_choices, source, review_status, execution_verified. 모델에는 request/context와 카탈로그 명세만 넣는다.
- `tools.json`: 고정 catalog `hermes-ten-tools-2289142ab57a`와 바이트 동일한 명세.
- `ai-proposals.json`: `{case_id: [{proposer, disposition, decision, acceptable_choices, confidence, rationale}]}`. AI의 제안·보류·제외를 사람 정답과 분리한다. confidence는 비어 있지 않은 문자열이며 숫자 원본은 confidence_score로 함께 보존할 수 있다. action이 없는 도구에서는 action 필드를 생략한다.
- `leak-audit.json`: 조사 범위와 corpus fingerprint를 기록한 누수·출처 검사. 원문을 출력하지 않는다.
- `manifest.json`: schema_version=frozen-eval-v1, status=candidate|frozen, catalog_version, candidate_order_seed, sha256(상대 파일명→해시), leak_audit(status와 file), 출처·검토 종류별 건수, 목표와 부족분.

`sha256`은 cases.jsonl·tools.json을 포함하고 provisional-ai에는 ai-proposals.json도 포함한다. 채점에는 status=passed인 감사 파일이 해시에 포함돼야 한다. 파일 경로 이탈과 해시 변경·빈 입력·후보 밖 도구·중복·미해결 라벨은 거부한다. 채점은 status=frozen과 운영자가 고정한 manifest SHA가 일치할 때만 가능하다. candidate는 gate-only 검사를 할 수 있지만 평가 완료 상태가 아니다.

## 정답과 출처

`source.kind=real`은 실제 기록에서 가져왔다는 뜻이며 사람 작성·정답·실행 성공을 보증하지 않는다. user 역할에 주입된 시스템 알림·작업자·cron도 제외한다. source.actor_kind와 provenance, human_origin_verified, 문맥·개인정보 검토 상태를 별도로 둔다.

초기 decision=null, acceptable_choices=[], review_status=needs_human_review, execution_verified=false다. 관측된 호출이 있더라도 별도 필드에 두고 정답으로 승격하지 않는다. source는 실제/합성 자료를 명확히 구분한다. 라벨은 모델 예측을 보기 전에 도구 명세를 대조해 검토한다. 단일 후보로 해결할 수 없는 다단계·문맥 누락·카탈로그 밖 요청을 억지로 단일 라벨로 만들지 않는다.

`--labels human`은 human_reviewed와 명확한 select_tools 정답을 요구한다. `--labels provisional-ai`는 서로 다른 검토자 2명 이상의 label 제안이 같은 단일 도구에 합의해야 하며 human review로 승격하지 않는다. freeze는 파일 불변성이지 라벨 품질 인증이 아니다. AI 합의 결과는 탐색적 일치도로만 표시하며 사람 정답 기준 실제 정확도로 주장하지 않는다. 목표 미달 자료는 부분 집합으로 표시하고 real100으로 보고하지 않는다. 미해결 사례가 섞이면 전체 채점을 거부하며, 제외 내역은 manifest/감사 결과에 남긴다.

## 누수·개인정보와 재현

모든 기존 train/validation/test와 합성 보강·진단 자료를 비교한다. 정확·정규화 중복은 거부하고 유사 문장·같은 시나리오/대화 파생 관계는 별도 감사한다. 같은 세션의 여러 요청이 독립 표본이라는 주장을 하지 않는다. 선택지 순서는 라벨과 독립적으로 seed와 case_id로 정하며 두 모델에 동일하게 적용한다.

원문 DB·실제 요청·가명 키·AI 제안은 Git 제외 private 경로에 둔다. credential 줄·PII·파일명·알려진 신원을 정제하고 내부 별칭은 비공개 설정에서만 받는다. 짧은 이름과 대소문자 변형도 검토하며 준식별 개인 서술은 별도 보류한다. 정규식 정제가 익명성을 보증하지 않는다. 코드·합성 테스트·집계 보고서만 커밋하고 원문을 문서/터미널 출력/MLflow에 무검토 공개하지 않는다.

고정 head는 이전 regularized 선정 모델과 augmented 모델이며 학습·추가 튜닝 없이 같은 입력·명세·라벨로 한 번 비교한다. 두 모델의 학습 당시 설명은 서로 달랐으므로 순수 데이터 보강 효과의 인과 실험으로 해석하지 않는다. 비교를 실행하면 한 MLflow Parent의 두 Child로 기록한다. 후보 준비가 막힌 상태에서 빈 benchmark Run이나 가짜 성능 지표를 만들지 않는다.
