# MLflow 파라미터 추가 후 완료 검증 오류 복구

**복구 완료.** 학습이 끝난 Child 10을 재학습하지 않고 같은 Parent에서 완료 처리한 뒤 후속 학습을 이어갔다. 해당 공동 탐색은 이후 20회 모두 완료됐다.

## 원인

비교 파라미터 5개를 추가했지만, supervisor의 완료 검증은 기존 5개만 허용하고 있었다. 실제 파라미터가 10개가 되자 업로드 후 검증에서 종료됐다.

| 구분 | 확인 내용 |
|---|---|
| 화면 증상 | 공동 탐색 표가 더 이상 갱신되지 않음 |
| 실제 학습 | Child 10은 17에폭·2,550 steps 학습 완료 |
| 실패 단계 | 업로드 후 MLflow key 집합 검증 |
| 오류 | `Nonessential MLflow keys were logged, or required keys are missing` |

## 수정

[실행기](../../scripts/run_laya_full_experiment.py)의 `validate_logged_keys`에서 계획의 필수 파라미터를 검사하되, 승인된 비교 파라미터 5개는 추가로 허용하도록 수정했다.

필수 항목 누락, 알 수 없는 파라미터, 지표의 추가·누락은 계속 거부한다. 새 계획에서 비교 파라미터를 필수로 선언한 경우에도 누락을 허용하지 않는다.

## 복구 결과

| 항목 | 결과 |
|---|---|
| 수정 커밋 | `3ac42f4` |
| 회귀 검증 | 서버 tracking 테스트 14개 통과 · 기존 데이터 검증 통과 |
| Child 10 | 기존 실행에서 업로드·완료 검증만 재수행해 `FINISHED` |
| 재학습 여부 | 없음 · 기존 파일 6개의 SHA·mtime와 runner 목록 동일 |
| 후속 실행 | Child 11 시작 확인 · 이후 공동 탐색 20회 완료 |

복구 전·후 증거는 Parent의 `recovery/` Artifact에 보관했다. 기존 실행, 모델, 원본 설정은 유지했다.

## 당시 복구 명령

기존 supervisor 종료와 학습 결과 파일을 확인한 뒤, 같은 state로 실행했다. 완료된 Child는 건너뛰고 미완료 단계부터 이어간다.

```sh
TORCHINDUCTOR_COMPILE_THREADS=4 PYTHONPATH=src /home/persona-ai/envs/jery-cuda/bin/python scripts/run_laya_full_experiment.py \
 --python /home/persona-ai/envs/jery-cuda/bin/python \
 --model-config configs/training/laya-ten-tools-train600-joint20-model-20261021.json \
 --state-dir outputs/laya-ten-tools-train600-joint20-20261021/tracking --detach
```

Parent: `3b7fd13fe2f74146a520fa64d0c2305d`
복구한 Child: `f599948302284d26ab2a7a66173d2379`
증거 파일: `tracking/recovery-param-keys-before.json`, `tracking/recovery-param-keys-after.json`
