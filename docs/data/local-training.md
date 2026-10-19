# Mac mini M4 로컬 학습 실행

등록 MCP 5개·도구 96개의 `mcp-routing-v2`를 사용하는 로컬 탐색 실험이다. 데이터는 미검토 합성 정답이며 실제 사용 정확도를 나타내지 않는다. Mac mini M4 32GB에서 CPU 프로세스 최대 4개로 기준 모델과 유한한 검증 설정 조합을 비교한다. 데이터가 작으므로 완료까지 걸린 실제 시간을 기록하며, 같은 데이터를 주말 내내 반복하도록 시간을 채우지 않는다.

## 담당과 계약

- `dev-claude`: `src/jery/model/`, `src/jery/evaluation/`, `configs/model/`, 모델 테스트와 실행 결과.
- `dev-agy`: `apps/review/`, `tests/review/`, `docs/review/`의 독립 데이터 감사.
- `dev-codex`: 데이터 버전 고정, [예측 계약](../contracts/prediction-v1.md), 실행 래퍼와 통합 검증.

공통 입력은 요청·호출 이전 맥락·후보 명세만 포함한다. 학습 통계는 train만 사용하고, 설정과 모델 계열은 validation으로 정한 뒤 test를 보고한다. 오류 분석 이후 test 수치를 올리기 위해 같은 평가셋으로 재튜닝하지 않는다. 검증·평가 미등장 클래스와 도구별 불균형을 결과에 함께 기록한다.

## 실행

모델 담당 결과가 통합된 저장소 루트에서 실행한다. Python 3 표준 라이브러리를 사용한다.

```sh
python3 scripts/validate_dataset.py
python3 scripts/validate_mcp_dataset.py
PYTHONPATH=src python3 scripts/run_local_training.py \
  --output outputs/local-training/m4-weekend-001 \
  --timeout-hours 48 \
  -- python3 -m jery.model.cli \
  --config configs/model/mcp-routing-v2-baseline.json \
  --run-id m4-weekend-001 --workers 4
```

실행 래퍼는 macOS에서 `/usr/bin/caffeinate -i -s`를 학습 명령에만 붙여 유휴·시스템 절전을 방지한다. 시스템의 영구 절전 설정은 변경하지 않는다. 종료 시 절전 방지 프로세스도 끝난다. `--allow-sleep`으로 끌 수 있다. 학습 자체의 재시작·복원 기능은 모델 구현에 따르며, 래퍼는 실패한 실행을 자동 재시도하지 않는다.

기존 출력 경로를 덮어쓰지 않으므로 다시 실행하려면 새로운 `--output`과 `--run-id`를 사용한다. 데이터·설정을 바꾸면 별도 실험으로 기록한다. 명령 인자에 인증 정보나 비밀을 넣지 않는다. 외부 서비스 연결이나 모델 다운로드는 이 명령의 일부가 아니다.

## 기록과 중단

실행 디렉터리에 다음 파일을 남긴다.

- `training.log`: 자식 명령의 stdout/stderr. Python 출력 버퍼링을 꺼 진행 상황을 기록한다.
- `run.json`: 명령, 작업 경로, 시작·종료 시각, runner/child PID, 반환 코드, 실행 시간, 상태.

상태는 `starting`, `running`, `completed`, `failed`, `timed_out`, `interrupted`다. `completed`는 명령의 종료 코드 0만 의미하며 모델 정확도가 충분하다는 인증이 아니다. 학습 과정·지표·데이터 해시는 모델 CLI의 `outputs/model/mcp-routing-v2/<run-id>/report.json`과 `report.md`를 확인한다. 가중치는 `models/mcp-routing-v2/<run-id>/`에 저장하며 Git에 추가하지 않는다.

실행 중 `Ctrl-C` 또는 `kill -TERM <run.json의 runner_pid>`로 중단할 수 있다. 래퍼는 자신이 시작한 프로세스 그룹에 종료 신호를 보내고, 10초 내 끝나지 않으면 해당 그룹을 강제 종료한다. 타임아웃도 같은 방식으로 정리한다. OS 강제 종료·전원 손실로 래퍼가 기록하지 못했다면 `running`이 남을 수 있으므로 실제 PID의 생존 여부를 별도로 확인한다.

수치 연산 라이브러리의 스레드 환경 변수는 4로 설정하고 모델 CLI는 worker 4개, worker당 thread 1개 설정을 사용한다. 이는 CPU 사용률이나 메모리의 OS 강제 제한이 아니다. 큰 신경망 학습이나 GPU/Metal 학습은 현재 기준 실험에 포함하지 않는다.

실행 래퍼와 예측 계약 검증:

```sh
python3 -m unittest discover -s tests/integration -v
```
