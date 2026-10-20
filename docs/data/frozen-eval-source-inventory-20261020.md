# 새 실제 요청 평가셋 준비 — 2026-10-20

사용자가 기존 학습·튜닝에 쓰지 않은 실제 요청 약 100개를 Claude·Antigravity와 함께 확보하고 평가하도록 요청했다. 현재 실제 데이터가 목표보다 부족해 **평가셋 동결과 모델 평가는 하지 않았다**. 합성 사례를 실제 요청으로 채우지 않는다.

## 출처와 수집 경로

현재 접근 가능한 Hermes 프로필 DB를 read-only로 조사했다. root Discord user 기록은 2,050개였으나 active/uncompacted는 469개였다. 기존 collector가 허용하는 종료 root session은 1개뿐이어서 기존의 종료·히스토리 검사를 완화하지 않았다.

별도 standalone snapshot 경로를 구현했다. root Discord user·active1·compacted0인 요청 자체만 읽으며, 열린 세션도 과거 실행 재현이 아니라 독립 문장 후보로만 다룬다. context는 비워 두므로 이어 말하기·이전 문서 참조는 정답 검토에서 보류한다. 현재 호출의 결과나 이후 답변은 읽지 않고, 관측된 도구를 정답으로 쓰지 않는다.

최종 현재 private snapshot은 후보 13개다. 제외 집계는 rewritten/inactive 1,581, 길이 제한 55, 도구 연관어 없음 381, user 역할에 주입된 시스템 알림 19, 개인 서술 privacy quarantine 1이다. 연관어 검사는 라벨이 아니며 도구명을 직접 언급하는 요청에 편향된 선택이다. 실제 100개나 도구별 균형을 확보했다는 뜻이 아니다. 초기 초안은 private 경로에 보존하고 닉네임·시스템 알림 문제를 수정한 후 새 파일로 수집했다.

원문·가명 키·실제 후보·AI 제안은 Git 제외 0600 private 파일로 유지한다. Public 코드에 계정 이름이나 원문을 넣지 않는다. source DB나 Hermes 설정은 변경하지 않았다. 정규식 정제는 개인정보 완전 제거를 보증하지 않으며 출처/라벨/개인정보 검토가 별도로 필요하다.

## 역할과 검증

Codex는 source inventory·standalone 수집·신원 정제·기존 자료 비교·공통 계약을 맡았다. Claude는 해시·카탈로그·라벨·누수 게이트와 고정 head 평가 CLI를 구현하고 실제 후보 문맥·개인정보 문제를 교차 검토했다. Antigravity는 과거 자료와의 근접/정규화 중복·출처·개인정보 감사 및 별도 라벨 검토를 맡았다. Orca Run run_dcd112b846ce에서 실제 Claude/Antigravity 작업자를 사용했다.

Claude의 최초 14개 후보 검토에서는 단일 도구로 라벨을 제안할 수 있는 요청이 get_portfolio 1개뿐이었다. 문맥 필요 8, 조언/다단계 제외 3, 카탈로그 밖 2였다. 이 중 개인 서술 1개를 root가 최종13개 후보에서 보류했다. Antigravity는 최종 13개를 독립 검토해 단일 선택 1, 문맥 필요 5, 제외 5, 카탈로그 밖 2를 제안했다. 두 검토자의 단일 선택 합의는 동일한 get_portfolio 1개다. 나머지 12개는 단일 선택 평가에 사용할 수 없으며 일부 보류·제외 분류는 불일치한다. AI 적격 기준 목표까지 99개 부족하고 human_reviewed는 0개이므로 사람 검토 기준 100개 부족하다. 사용자는 추가 과거 로그 출처가 없다고 확인했다. 이는 AI 제안이며 human_reviewed가 아니다.

수집 테스트는 원본 바이트 보존, 시스템/작업자/압축/credential 제외, 기존 입력 비교, 신원 대소문자·짧은 이름·허용된 대괄호 보존, private 권한, 덮어쓰기/키 alias 거부를 검사한다. Source counts는 같은 원본 snapshot에 대한 집계이며 실제 평가 성공 지표가 아니다.

## 재현

실제 DB 경로·private 키 경로를 로컬 환경에 맞춰 지정한다. private 키 JSON의 redactions에는 공개할 수 없는 신원 목록을, privacy_exclude_ids에는 추가 보류할 가명 case ID를 설정할 수 있다. 내부 신원 값은 문서·Git에 기록하지 않는다.

```sh
python3 scripts/collect_frozen_eval_candidates.py \
  --source /private/hermes/profile/state.db \
  --catalog datasets/hermes-ten-tools-v2/tools.json \
  --output data/local/frozen-eval-new/candidates \
  --key-file data/local/frozen-eval-new/private-key.json
PYTHONPATH=src python3 -m unittest discover -s tests/data -p test_frozen_eval_collection.py
```

공통 동결 계약은 [frozen-eval-v1](../contracts/frozen-eval-v1.md)을 따른다. 현 스냅샷은 candidate 상태, 미해결 gold이며 모델 평가 승인용 manifest가 아니다. 추가 실제 요청과 라벨 검토가 필요하다. 합성 fixture 테스트와 real benchmark를 구분한다.

## 통합 완료와 현재 중단 지점

전체 테스트 168개가 통과했고 `python3 scripts/validate_dataset.py`는 130 examples / 10 tools / 65 isolated scenarios PASS였다. Antigravity의 1,484개 과거 자료(진단 포함)에 대한 정확·정규화·근접 중복 검사에서 중복 0개, 정제·입력 경계 검사 위반 0개였다. 이는 검사 범위에서의 결과이며 개인정보 완전 제거 보증은 아니다.

원본 AI 제안은 별도 보존했다. 통합 private 제안에서는 숫자 confidence를 원래 score와 함께 문자열로 보존하고, action이 없는 도구의 null action을 생략해 공통 예측 계약에 맞췄다. 정답 의미나 검토자 결론은 변경하지 않았다. 최종 candidate-final/cases.jsonl의 SHA-256은 b4d65338621c87a98600658a77e21514dbd03ec0a9ed8b840db7d7408e2ccdff로 감사한 v4와 동일하다.

실제 private 패킷 gate-only 검사는 human에서 미해결 13개, provisional-ai에서 미해결 12개를 거부했다. 동결/manifest 승인 없는 scoring도 모델 로드 전에 거부했다. 최종 상태는 candidate이며 모델 추론·재학습·새 평가 Run을 실행하지 않았다. GPU 평가 CLI는 합성 테스트와 차단 동작까지 검증했으며 실제 head 비교 실행은 검증하지 않았다.

Codex의 수집/통합, Claude의 평가 게이트/검토, Antigravity의 독립 감사/검토 작업은 완료했다. 추가 실제 요청 부족으로 100개 평가셋 완성은 남아 있다. 다음 경로는 앞으로의 실제 요청을 같은 기준으로 누적하는 것이다. 사용자가 새로 작성한 테스트 요청은 authored로 별도 표시해야 하며 실제 관측 로그와 섞어 real100이라고 부르지 않는다. 원문·가명 키·검토 제안은 로컬에 유지하고 MLflow에는 준비 상태 집계 JSON만 기록한다.
