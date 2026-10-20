# 10개 도구 학습 보강 원문 v1

180개 AI 작성 합성 사례(도구당 18개), 60개 시나리오(도구당 6개)다. 각 시나리오는 직접 요청·바꿔 쓴 요청·이전 맥락으로 해결되는 요청의 3개 형태로 모두 train에 둔다. 사람이 검토한 정답이나 실행된 로그가 아니다. 모의 주문 문장도 데이터이며 실제 도구를 호출하지 않았다.

기준은 기존 Hermes 카탈로그 `datasets/hermes-ten-tools-v1/tools.json`의 고정 명세다. frontmatter 수정·파일 이름 변경·.base 뷰·단일 종목 보유 현황·일별 종가·계좌 전체 손익 등 기능 경계를 보강한다. 검증 오류 유형을 관찰한 뒤 작성했으므로 validation-informed 합성 보강이다. 기존 validation/test의 문장을 학습으로 이동하지 않는다. 정규화된 중복 검사와 그룹 검사가 재생성 과정에 포함된다.

`review_status=needs_human_review`, `execution_verified=false`를 유지한다. `action`은 카탈로그 enum과 대조하며 action 필드가 없는 Swallow 함수에는 null을 쓴다. 입력 특징은 request/context뿐이며 정답·시나리오 ID·검토 정보는 특징에 넣지 않는다. 사람 검토 및 독립적인 실제 요청 평가가 남아 있다.
