# Hermes 고정 10개 도구 v2 — 기능 설명과 학습 보강

기존 v1 파일과 source export는 변경하지 않았다. 같은 도구 명세를 유지하고 짧은 선택지 설명을 `configs/training/laya-ten-tools-criteria-v2-20261020.json`으로 명시적으로 확장했다. 새 catalog_version은 manifest를 따른다. decision-export-v2의 단일 10개 선택 의미는 그대로다.

- description_control: 기존 train 60개에 새 설명만 적용.
- augmented: 같은 60개와 신규 train 180개를 합쳐 240개(도구당 24개).
- 두 profile의 validation/test 각 20개는 바이트 동일하다.
- 기존 v1 validation/test의 request/context, gold, alias 배치, scenario_id, 행 순서는 그대로 유지한다. 선택지 설명과 catalog provenance가 바뀌므로 v1과 바이트 동일하거나 완전히 같은 입력이라고 주장하지 않는다.

원문 보강 사례는 `datasets/hermes-ten-tools-contrast-source-v1/cases.jsonl`에 두고 파생 export는 Git 제외 `outputs/laya-ten-tools-contrast-20261020/`에 생성한다. `scripts/build_ten_tool_contrast.py`로 재현하며 기존 경로 덮어쓰기를 거부한다. 원본과 신규 출처 해시는 manifest에 보존한다.

`label-inspection.json`은 검증 오류 10건의 AI 카탈로그 대조 결과다. 기존 라벨 변경은 없으며 사람이 검토한 정답으로 승격하지 않는다. 검증 20건과 test 20건은 이미 반복 관찰된 회귀 자료다. 새로운 독립 평가나 실제 사용자 성능 근거가 아니다. 정답 검토와 실제 요청 수집이 남아 있다.
