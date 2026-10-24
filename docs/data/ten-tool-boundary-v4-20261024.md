# 도구 경계 데이터 v4 · 학습800개와 설명 개선

기존 학습600개에 새 합성200개를 추가했다. 도구마다80개로 수량을 맞추고, 읽기/수정·조회/주문·과거/현재 가격·Bases/Dataview 경계를 구분하도록 설명을 개선했다. 이번 작업은 데이터 준비이며 새 모델 학습이나 성능 측정은 실행하지 않았다.

## 저장 구성

[MLflow 데이터 Parent](https://mlflow.dove-nest.com/#/experiments/47/runs/ff162f7af7d64c05bd2a9fde44d9d8bd)

실험은 `Jery`, Parent 이름은 `laya-ten-tools-dataset-v4-20261024`다. 각 Child는 데이터 준비 기록이며 학습 완료 Run이 아니다. 성능 지표나 epoch를 만들지 않는다.

| Child | Train | Validation | Test | 설명 |
|---|---:|---:|---:|---|
| baseline | 600 | 20 | 20 | 기존 |
| description_only | 600 | 20 | 20 | 개선 |
| data_only | 800 | 20 | 20 | 기존 |
| combined | 800 | 20 | 20 | 개선 |

각 Child의 `data/train.json`, `data/validation.json`, `data/test.json`은 실제 데이터 전체를 담은 JSON 배열이다. `data/native-export.zip`은 기존 실행기가 읽는 JSONL 세 파일과 manifest를 묶은 재현용 자료다. `catalog/tools.json`, `catalog/criteria.json`, native 형식 감사도 함께 저장한다. JSON 배열은 MLflow Dataset 입력으로 등록해 split·digest·원본 Artifact 위치를 연결한다. 이 등록 방식은 [MLflow Log Inputs API](https://mlflow.org/docs/latest/api_reference/rest-api.html)에 따른다.

params는 profile·dataset_version·catalog_version·train_rows만 기록한다. 상세 건수·검증·해시·정책은 JSON Artifact로 보관한다. Parent에는 추가200개 원문, 이전/새 설명, 데이터 및 감사 manifest와 업로드 검증 내역을 남긴다. Artifact는 업로드 후 다시 내려받아 SHA-256을 비교한다.

## 변경과 보존

- 신규200개: 한국어 합성, 120개 시나리오, 24개 문맥 사례. 작성자는 Codex이고 독립/사람 검토를 완료했다고 표시하지 않는다.
- 기존 요청·맥락·정답·alias를 보존했다. 기존 설명 비교군의 validation/test는 바이트도 동일하다. 설명 개선 비교군은 criteria와 카탈로그 연결만 갱신했다.
- 도구 ID·원래 설명·input_schema를 보존하고 모델용 `routing_description`을 새 카탈로그에 추가했다. export criteria와 동일한 내용이다.
- 신규 카탈로그는 `hermes-ten-tools-0b320e20bf3f`; 기존 설명 비교군은 `hermes-ten-tools-2289142ab57a`다. 라벨·`decision-export-v2` 계약은 유지한다.

예를 들어 vault는 ‘메모 읽기·검색·생성·파일관리. 본문 수정 제외’, edit는 ‘기존 메모 본문·제목·속성 수정. 조회 제외’로 구분했다. portfolio는 계좌 전체 현황 조회, order는 가상 매매 주문 실행으로 표현했다. 실제 도구 호출이나 주문은 수행하지 않았다.

## 검증 결과

데이터 테스트37개, 기존 데이터 검증130개 사례/10개 도구/65개 시나리오가 통과했다. 새800개 구성은 도구당80개, 전체475개 시나리오다. 동일 입력으로 재생성한 export 해시가 같고, 원본 파일 보존·분할 간 시나리오/입력 중복·후보/정답/action 매핑을 검사했다.

원문200개를 기존 train600·validation20·test20과 비교했을 때 정규화 중복과 문자3-gram/단어 Jaccard 0.55 이상 근접 중복은 없었다. 초안 중 기존 validation과 유사한 Dataview 상태 요청 하나는 다른 DQL 문법 검사 사례로 교체했다. 이 검사가 의미상 독립성을 보증하지는 않는다.

실제 회사 서버의 Laya tokenizer/native builder로 네 구성 총2,960개 입력을 확인했다. skipped/truncated/feature mismatch/target mismatch는 모두0개다. 개선 설명의 최대 sequence382 tokens, 기존 설명358 tokens다. 초안 설명 일부가 후보별24-token 한도를 초과했으므로 설명을 줄여 다시 검증했다. 모델의 max_len1024/head_max_len256은 유지했다.

## 재현과 다음 학습의 조건

전체 명령은 [v4 README](../../datasets/hermes-ten-tools-v4/README.md)에 있다. 생성기는 기존 [build_ten_tool_contrast.py](../../scripts/build_ten_tool_contrast.py)를 재사용하고 새 카탈로그에 routing_description을 넣는 선택 동작만3줄 추가했다. 새 의존성은 없다.

새 대조 쌍80개는 같은 scenario 안에 서로 다른 두 라벨이 있다. 분할 시 이 쌍을 분리하면 안 된다. 기존 homogeneous-only 5-fold 생성기는 혼합 라벨 그룹을 거부하므로 새800개에 직접 재사용하지 않는다. 다음 학습 전에 일반 grouped stratification을 쓰거나, 기존600개 fold를 유지하고 새200개를 train-only 보강으로 넣는 계획을 정해야 한다. 후자의 경우 새200개는 OOF 평가 대상이 아니며 이를 명시해야 한다.

기존 validation/test는 여러 번 관찰한 회귀 자료다. 실제 독립 test 준비는 여전히 후보13개·사람 검토0개이며, 이 합성200개를 실제 test200으로 대신하지 않는다. 학습 정확도 향상 여부는 앞으로 네 구성의 통제 비교로 확인한다.
