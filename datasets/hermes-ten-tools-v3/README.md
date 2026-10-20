# Hermes 고정 10개 도구 v3 — 학습 데이터 보강

v2의 train 240개에 Claude·Antigravity가 새로 작성하고 상호 검토한 합성 train 360개를 더하는 학습 데이터 버전이다. 최종 train은 600개(도구당 60개)이며 검증·평가는 각 20개다. 건수·검증 결과는 manifest, readiness.json과 통합 보고서를 따른다. 원본 보강 자료는 `datasets/hermes-ten-tools-contrast-source-v2/`다.

카탈로그 `hermes-ten-tools-2289142ab57a`, 짧은 선택지 설명 v2.1-native-budget, seed 3407과 decision-export-v2 의미는 유지한다. 같은 설명을 사용하는 v2 augmented export를 기점으로 validation/test 각 20개와 기존 train 240개의 바이트를 보존한다. 새 파생 데이터는 기존 디렉터리를 덮어쓰지 않고 아래 명령으로 만든다.

```sh
python3 scripts/build_ten_tool_contrast.py \
  --base outputs/laya-ten-tools-contrast-20261020/augmented \
  --catalog-dir datasets/hermes-ten-tools-v2 \
  --authored datasets/hermes-ten-tools-contrast-source-v2/cases.jsonl \
  --dataset-version hermes-ten-tools-v3 \
  --dataset outputs/reproduce-training-v3/catalog \
  --output outputs/reproduce-training-v3/exports
```

처음 복제한 저장소에서는 기존 v1 export와 v2 export를 먼저 재현해야 한다. 기존 빌더와 README의 명령을 사용하고, catalog 디렉터리가 이미 존재하면 새 outputs 하위에 재생성한다. source catalog/manifest 해시는 같은 내용으로 재현돼야 한다. 새 경로로 두 번 재생성한 결과의 해시를 비교할 수 있다.

새 export의 `description_control`은 기존 train 240개, `augmented`는 보강 train 600개이며 각 profile의 validation/test는 v2 augmented와 동일하다. 프로필 이름은 기존 빌더와 호환하기 위해 유지하며 v3에서 설명을 바꿨다는 뜻은 아니다.

모든 정답은 `needs_human_review`, 실행 검증은 false다. AI 상호 검토가 사람 정답이나 실제 실행 검증을 대신하지 않는다. 이번 작업은 학습 데이터 제작이며 추가 학습·모델 성능 평가는 실행하지 않는다. 기존 validation/test는 반복 관찰된 회귀 자료로 실제 사용자 정확도 근거가 아니다. 모의 주문을 포함한 모든 문장은 합성 데이터이며 도구를 실제 호출하지 않는다.

최종 export는 `outputs/laya-ten-tools-training-v3-20261020/final-exports/augmented/`에 있다. 신규 360개는 Claude·Antigravity가 작성하고 Claude·Codex가 교차 검토했다. Antigravity 상대 검토는 quota 제한으로 미완료라 별도 기록했다. 전체 테스트 169개 및 native builder 640개 입력 검사에서 skipped/truncated/feature mismatch/target mismatch가 모두 0개다. 이 결과는 형식·정답 매핑 검증이며 모델 성능 향상 측정이 아니다.
