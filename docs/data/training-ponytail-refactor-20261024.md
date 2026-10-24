# 학습 실행 코드 정리 · Ponytail

탐색 후보 생성과 CV 설정 전달을 기존 함수 안에서 정리했다. 새 의존성이나 실행 프레임워크는 추가하지 않았다. 이번 작업에서 새 학습은 실행하지 않았다.

## 변경 내용

| 대상 | 변경 | 효과 |
|---|---|---|
| 후보 생성 | 모든 기존 후보와 비교해 중복 제외 | 같은 seed·같은 조건의 반복 학습 방지 |
| 고정 LR 탐색 | TPE가 중복을 제안하면 미사용 조합에서 seed로 선택 | 탐색이 한 조합에 몰려도 새 조건 평가 |
| 후보 소진 | 남은 조합이 없으면 명시적 오류 | 중복 재실행·무한 재시도 방지 |
| CV 설정 | `fold_trial_settings`에서 plan·model·tracking 값 함께 생성 | 샘플링한 값이 실제 fold 학습과 표에 동일하게 전달 |
| 입력 검사 | 탐색 항목 목록 공유, 잘못된 값·중복 선택지·알 수 없는 모드 거부 | 실행 전 설정 오류 발견 |
| 반복 코드 | 파라미터 집합 재검사·설정 재계산·미사용 인자 제거 | 호출 흐름 단순화 |

고정 LR의 미사용 조합 선택은 순수 TPE 제안과 구분해 후보 JSON의 `proposal=unseen_grid`에 남긴다. 첫 기준 조건은 `reference`, 일반 제안은 `tpe`다. 현재 864개 조합은 중복 발생 시에만 열거한다. 훨씬 큰 탐색 공간에는 별도 메모리 제한 또는 제한된 표본 추출이 필요하다.

## 검증

- 서버의 분리된 임시 폴더에서 tracking 15개·CV 6개·계획 6개, **총 27개 통과**.
- 실제 Hyperopt 20개 후보: 기준 첫 조건 유지, LR 고정, 서로 다른 조합 20개, test 점수의 탐색 배제 확인.
- 강제로 중복 제안을 넣어 미사용 조합 선택과 후보 소진 처리를 확인.
- 완료된 실제 실험의 이력 19개 구간을 다시 입력: 중복 제안 0개, 미사용 조합 선택 6회. 원본 state SHA 보존 확인.
- 기존 데이터 검증 통과: 130개 사례 · 10개 도구 · 65개 시나리오.

```sh
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_cv_experiment.py
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_full_tracking.py
PYTHONPATH=src python -m unittest discover -s tests/integration -p test_cuda_experiment_plan.py
```

Hyperopt가 설치된 기존 학습 환경에서 실행한다. 로컬 환경에서 Hyperopt가 없으면 관련 테스트는 건너뛰므로, 위 27개 결과는 서버 환경 기준이다.

## 다음 작업 제안

| 우선순위 | 작업 | 판단에 필요한 결과 |
|---|---|---|
| 1 | 기준 조건과 최고 조건의 OOF 오류 비교 | 600개 사례에서 도구별 혼동, 반복 오답, 정답 라벨 오류 분리 |
| 2 | 개선 조합의 seed 반복 검증 | 같은 fold에서 기준·최고 설정을 3개 seed로 비교해 개선 일관성 확인 |
| 3 | 새 독립 test 셋 확보 | 기존 사례의 표현 변형과 분리하고 사람이 검토한 실제 요청 평가 |
| 4 | 전송 오류 처리 범위 정리 | prepare 단계에도 supervisor와 같은 재시도 정책 적용 검토 |

현재 근거는 기준 validation 정확도 74.83% → 최고 76.83%, F1 0.7449 → 0.7643이다. 5개 fold 중 3개만 개선됐으므로 파라미터 탐색 확대 전에 오류와 개선의 일관성을 확인하는 편이 낫다. 기존 test는 20개이며 반복 관찰했으므로 최종 일반화 성능의 근거로 쓰지 않는다.

독립 test는 우선 도구당 20개, 총 200개 수준을 준비하는 안을 제안한다. 이는 준비 목표이며 충분한 통계적 정밀도를 보장하는 기준은 아니다. 모델과 설정 선택을 마친 뒤 평가한다.

## 코드

- [공통 후보 생성·tracking 설정](../../scripts/run_laya_full_experiment.py)
- [CV 조건 구성·실행](../../scripts/run_laya_cv_experiment.py)
- [계획 검증](../../scripts/prepare_laya_cuda_experiment.py)

기존 데이터·모델 입출력 계약은 유지한다. 완료된 실험의 후보와 점수는 변경하지 않는다. 새 후보 생성부터 중복 방지 정책이 적용된다.
