# 고정 10개 도구 학습 데이터 보강 — 2026-10-20

학습 분할을 240개에서 600개(도구당 60개)로 보강했다. 검증 20개·평가 20개, 카탈로그·선택지 설명·선택지 순서 seed는 유지했다. 이번 작업은 데이터 제작이며 새 학습이나 모델 성능 측정은 실행하지 않았다.

## 데이터와 검토

신규 합성 360개(도구당 36개, 260개 시나리오, 문맥 요청 54개)는 Claude·Antigravity가 각각 180개씩 작성했다. 원문은 `datasets/hermes-ten-tools-contrast-source-v2/cases.jsonl`, 사례별 검토는 같은 폴더의 `reviews.json`, 출처·검토 해시는 manifest.json에 있다. 전체 train은 355개 시나리오다. 현재 고정 10개 후보 중 단일 도구 선택만 다루며 action 및 no_tool/clarify/unsupported 예측을 학습시키는 자료는 아니다.

Claude는 Antigravity 초안 180개에서 28개에 수정을 요청했다. 이전 버퍼를 가정하는 요청, 답을 반복하는 assistant 확인, 동일 템플릿에서 종목·숫자만 바꾼 문장, action 표기와 경로 공백을 보완했고 수정본 180개를 다시 검토해 모두 AI accept로 판정했다. 원본 대비 변경된 요청과 metadata도 대조했다. Antigravity는 수정 후 individual quota 제한에 걸려 Claude 작성분 교차 검토와 최종 commit/push를 완료하지 못했다. 해당 Dispatch는 실패로 기록하고 정확한 수정본과 미커밋 diff를 보존했다. Codex가 Claude 작성분 180개를 직접 읽고 라벨·명세·사전 문맥을 대조해 상대 검토를 대신했다. 호가나 장중 고가·저가로 읽힐 수 있는 요청 2개, 속성 키가 모호한 수정 요청 1개를 명확히 했다. 라벨은 바꾸지 않았다.

Orca Run `run_d483ef3251c5`의 실제 Claude·Antigravity 작업자를 사용했다. Claude 작성·검토 커밋은 60775c6, c00f015, 2e787a9, df59f8e로 dev-claude에 푸시됐다. Antigravity 초기 작성 6031f74는 dev-agy에 푸시됐지만 최종 수정본은 해당 브랜치에서 미커밋 상태로 보존하고 dev-codex의 데이터 원문에 통합했다. 최종 수정본 SHA는 8d44816f8543f354e4be27c074d0700d156a0e02428fc18299de0ac7cd24e715다. 완료하지 못한 상대 검토를 Antigravity 검토로 표기하지 않았다.

모든 사례의 review_status는 needs_human_review, execution_verified는 false다. AI accept는 사람 정답·실제 실행 검증이 아니다. 실제 Hermes 후보 13개는 사용하지 않았다. 중립 assistant 문구와 '말고' 대비 표현 등 합성 문체 편향은 남아 있다.

## 보존과 검증

기존 `build_ten_tool_contrast.py`를 재사용했다. 같은 카탈로그일 때 기존 행을 바꾸지 않고 새 train만 덧붙이며 `--dataset-version`으로 데이터 버전만 바꾼다. v2의 파일이나 원문을 덮어쓰지 않는다. 결과의 description_control은 기존 train240, augmented는 train600이다. 이 프로필 이름은 기존 빌더 호환용이며 v3에서 설명을 바꿨다는 뜻이 아니다.

- 전체 unittest 169개 통과. 기존 데이터 검증: 130 examples / 10 tools / 65 isolated scenarios PASS.
- 서로 다른 새 경로에 두 번 재생성한 모든 export 파일이 동일했다.
- 기존 train240 prefix, validation/test 파일 바이트와 해시를 보존했다.
- source ID·시나리오 분할·정규화 중복·고정10개 후보와 gold 매핑 검사를 통과했다. 신규 학습 요청과 현재 heldout의 3-gram/단어 Jaccard 0.55 이상 후보는 0개였다. 유사도 검사가 의미상 누수를 완전히 보증하지는 않는다.
- 최종 360개 원문과 360개 AI accept의 ID·기대 도구·action이 일치한다.
- 회사 CUDA 환경에서 모델 가중치를 읽거나 학습하지 않고 native builder/tokenizer로 600/20/20개를 검사했다. skipped/truncated/feature mismatch/target mismatch=0, 최대 sequence 358/269/272 tokens, max_len1024·head_max_len256이었다.

카탈로그 hermes-ten-tools-2289142ab57a, 표현 catalog-grounded-criteria-v2.1-native-budget, decision-export-v2와 seed3407을 유지했다. 빌더는 Python 표준 라이브러리를 사용하며 새 의존성은 없다. 전체 테스트에는 기존 requirements-model 환경을, native 검사에는 Unsloth2026.10.6 / Laya0.4.9 / Transformers5.17.0 환경을 사용했다. native 형식 통과는 원래 Laya 사전학습 코퍼스와의 분포 일치나 성능 향상 증거가 아니다.

## 위치와 재현

학습 export: `outputs/laya-ten-tools-training-v3-20261020/final-exports/augmented/`. 원문·카탈로그·manifest·검토·작은 검증 집계는 Git에 보관하고 파생 export는 Git 제외 경로에 둔다. 모델 가중치와 대형 결과를 Git에 추가하지 않는다. 재생성 명령은 `datasets/hermes-ten-tools-v3/README.md`를 따른다. 기존 v1/v2 export가 없는 새 checkout에서는 두 기존 빌더를 먼저 새 outputs 경로에 실행한다.

```sh
PYTHONPATH=src python3 -m unittest discover -s tests/data -p test_ten_tool_contrast.py -v
python3 scripts/validate_dataset.py
# 기존 Unsloth CUDA 환경에서 가중치/forward 없이 수행
PYTHONPATH=src python scripts/audit_laya_export_format.py \
  --config /path/to/native-audit-config.json --output /new/path/native-readiness.json
```

현재 validation/test는 이미 반복 관찰된 합성 회귀 자료다. 보강 데이터의 실제 성능은 아직 측정하지 않았다. 후속 학습 시 필수 params/metrics와 나머지 JSON artifacts를 구분하고 같은 Parent의 Child로 기록하는 기존 정책을 따른다. 기존 MLflow 모델과 학습 결과를 새 데이터로 학습한 것처럼 변경하지 않는다.
