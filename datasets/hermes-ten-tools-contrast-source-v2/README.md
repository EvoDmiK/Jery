# 고정 10개 도구 학습 보강 원문 v2

신규 합성 train 360개(도구당 36개, 260개 시나리오)를 Claude와 Antigravity가 각각 180개씩 작성했다. Antigravity의 180개는 Claude가, Claude의 180개는 Codex가 카탈로그와 대조했다. 최종 채택 건수와 입력/검토 해시는 manifest.json을 따른다. 이 폴더는 학습 보강 원문이며 실제 로그나 평가셋이 아니다. 기존 v1 원문 180개는 변경하지 않는다.

`cases.jsonl`은 도구별 균형을 맞춘 원문 사례다. 도구 ID·action·출처·rationale·검토 상태는 모델 특징으로 사용하지 않고 request/context만 특징으로 쓴다. 직접 요청, 도구명을 명시하지 않은 일상 표현, 사전 문맥으로 대상이 확정되는 요청을 포함한다. 같은 시나리오의 표현 변형은 같은 scenario_id로 묶고 전부 train에 둔다.

`reviews.json`은 입력 SHA와 사례별 AI 검토자, accept/revise/exclude 판단, 기대 도구/action, 사유를 기록한다. 사람 정답을 대신하지 않으며 원문 review_status=needs_human_review / execution_verified=false를 보존한다. 수정이 필요했던 사례는 최종 원문과 다시 대조해야 한다.

명세에서 지원하지 않는 Bases evaluate, 주문 수량, RSI 이외 전략, 임의 시간별 주가, 계좌 초기화·관심종목 관리 등을 단일 선택 정답으로 넣지 않는다. 작업 결과를 보거나 호출 결과를 읽은 뒤 정답을 생성하지 않는다. 모의 주문 문장도 학습 데이터이며 실제 주문을 실행하지 않는다.

이 원문은 기존 v2 train240에 추가해 v3 train600을 만든다. 이전 검증/평가 문장, 정답, 명세, 선택지 순서와 파일 바이트는 유지한다. 정상적인 합성 정답이 실사용 성능을 보증하지 않는다. 상호 검토 및 기계 검증의 구체적 결과와 한계는 manifest 및 docs/data 통합 보고서를 따른다.

Antigravity는 자기 작성분 수정 후 사용량 한도에 걸려 상대 검토를 완료하지 못했다. 미완료를 성공으로 보고하지 않았고 Codex가 상대 작성분 180개를 이어서 검토했다. Claude는 Antigravity 초안의 28개 수정사항과 수정본의 모든 변경을 다시 확인했다. Codex는 Claude 원문의 3개 표현을 지원되는 기능·정확한 속성 키로 명확히 했다. 최종 각 사례의 도구/action 및 accept 판정은 reviews.json에서 원문과 대조할 수 있다. 문맥 사례는 54개다.
