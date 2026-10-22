# MLflow HTTP 520 · 업로드 중단 복구

16번째 LR 조건의 세 번째 fold가 학습·평가를 마친 뒤, MLflow Artifact 업로드에서 HTTP 520을 받아 전체 실행이 중단됐다. 모델 계산 실패가 아니라 기록 전송 실패다.

## 원인과 수정

| 구분 | 내용 |
|---|---|
| 중단 위치 | Trial 16 · fold 3 · 24에폭 학습 후 업로드 |
| 보존된 결과 | 학습 report, 평가 지표, 체크포인트 |
| 원인 | 재시도 대상에서 프록시 HTTP 520이 빠져 있었음 |
| 수정 | 두 supervisor가 같은 판별 함수를 사용해 520~524를 재시도 |
| 유지한 검사 | 400·401·403·404 등 클라이언트 오류와 데이터 검증 오류는 즉시 보고 |

[전체 학습 실행기](../../scripts/run_laya_full_experiment.py)와 [CV 실행기](../../scripts/run_laya_cv_experiment.py)의 재시도 정책을 통일했다. 학습 설정·데이터·LR 후보는 변경하지 않았다.

## 복구 결과

**2026-10-22 09:58 KST 확인:** fold 3의 업로드와 완료 검증을 마쳤고, MLflow에서도 `FINISHED`를 확인했다. 같은 Trial 16의 fold 4가 시작됐다. 기존 결과 파일 48개의 SHA·수정 시각과 기존 Parent·fold run ID가 모두 동일했다.

## 복구 절차

1. 두 supervisor의 lock이 해제되어 있고 기존 학습 프로세스가 없음을 확인했다.
2. Trial 16의 기존 결과 파일 48개에 대한 SHA·수정 시각과 fold별 run ID를 보관했다.
3. 수정 코드 `764a783`을 서버에 반영하고 기존 Parent·state로 재개했다.
4. 완료된 15개 조건과 기존 fold는 재학습하지 않고, 미완료 업로드부터 이어간다.

[기존 MLflow Parent](https://mlflow.dove-nest.com/#/experiments/47/runs/675bf807c9674425ba6f9de1ad536b91) 아래에서 재개했다. 기존 Child를 새로 만들지 않는다.

## 검증

- 서버 tracking 테스트 **15개**, CV 테스트 **4개** 모두 통과.
- 기존 데이터 검증 통과: 130개 사례 · 10개 도구 · 65개 분리된 시나리오.
- 복구 증거: 서버 상태 디렉터리의 `recovery-http520-before.json`, `recovery-http520-after.json`.

새 의존성은 없다. 실행 재개는 전체 20개 조건의 완료를 의미하지 않는다.
