# Gemma 4 E4B 로컬 실행 확인

Mac mini M4, 통합 메모리 32 GiB에서 Gemma 4 E4B 4비트 모델을 MLX로 실행한다.
미세조정 없이 요청, 호출 전 맥락, 후보 명세만 입력한다. 생성된 선택은 공통
prediction-v1 계약으로 검증하며 실제 MCP 도구는 호출하지 않는다.

## 모델과 환경

- 모델: `mlx-community/gemma-4-e4b-it-4bit`
- 다운로드 revision: `475b9088d29754a3379866cf5aeb6b41acd313c2`
- 가중치와 토크나이저: `models/gemma-4-e4b-it-4bit/` (약 4.8 GiB, Git 제외)
- 실행 환경: Python 3.12.15, MLX 0.32.4, mlx-lm 0.32.2, Transformers 5.20.1
- 입력: `datasets/mcp-routing-v2/validation.jsonl`, `tools.json`
- 카탈로그: `hermes-mcp-57e701d87790`; 출력 계약: [prediction-v1](../contracts/prediction-v1.md)

MLX LM의 Gemma 4 텍스트 로더를 사용하며 이미지·음성 입력은 시험하지 않는다.
thinking은 끄고 greedy decoding, 최대 생성 256토큰으로 실행한다.

## 재현

Apple Silicon에서 Python 3.12 가상환경을 만든 뒤 다음을 실행한다.

```sh
.venv/bin/python -m pip install -r requirements-llm.txt
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/gemma-4-e4b-it-4bit', revision='475b9088d29754a3379866cf5aeb6b41acd313c2', local_dir='models/gemma-4-e4b-it-4bit')"
python3 scripts/validate_dataset.py
python3 scripts/validate_mcp_dataset.py
caffeinate -i .venv/bin/python scripts/run_llm_smoke.py
```

`uv`로 만든 가상환경처럼 pip가 없으면 설치 줄 대신
`uv pip install --python .venv/bin/python -r requirements-llm.txt`를 사용한다.

검증 분할에서 후보가 최대 12개인 서로 다른 시나리오를 결정 유형별 최대 2개씩
파일 순서대로 고른다. 현재 기본 설정은 도구 선택 2건, no_tool 1건, clarify 1건,
unsupported 2건으로 총 6건이다. 후보 제한과 시나리오 중복 배제로 실제 건수가 줄어든다.
전체 96개 후보나 전체 평가 분할에 대한 벤치마크는 아니다.

결과는 `outputs/gemma-e4b-smoke/YYYYMMDD-HHMMSS/`에 저장한다.

- `system.txt`: 실제 사용한 시스템 프롬프트
- `predictions.jsonl`: 원본 생성 텍스트, 계약 검증, 결정·도구 일치, 시간
- `report.json`: 집계, 환경 버전, 데이터·프롬프트 해시, MLX 최대 메모리

시간은 프롬프트 처리와 생성을 포함한다. 첫 사례에는 컴파일 준비 비용도 포함된다.
MLX 최대 메모리는 운영체제 전체 메모리 사용량이 아니다. 계약을 위반한 출력은 실패로
계산하며 자동으로 수리하지 않는다. 도구 일치 지표는 action 정확도를 포함하지 않는다.

## 해석 범위

2026-10-17 실제 실행 결과:

| 실행 | 계약 유효 | 계약 유효 및 결정 일치 | 도구 선택 일치 | 응답 중앙값 |
|---|---:|---:|---:|---:|
| 첫 영어 프롬프트 | 2/6 | 2/6 | 0/2 | 5.084초 |
| 수정 한국어 프롬프트 | 5/6 | 2/6 | 0/2 | 5.109초 |
| 기존 선형 모델, 동일 사례 | 6/6 | 4/6 | 2/2 | 준비 후 1.890ms |

첫 실행은 `outputs/gemma-e4b-smoke/20261017-071522/`, 수정 실행과 선형 모델
비교는 `outputs/gemma-e4b-smoke/20261017-071618/`에 보존했다.
선형 모델 비교 원본은 수정 실행 폴더의 `baseline-comparison.json`이다.
LLM 중앙값은 전체 6건, 선형 모델 준비 후 중앙값은 최초 모델 로딩이 포함된 첫 건을
제외한 5건이다. 선형 모델 첫 호출은 로딩 포함 0.303초였다.

수정 실행의 모델 로딩은 3.186초, MLX 최대 할당 메모리는 약 5.034GB였다.
입력은 후보 4~8개, 1,515~4,307토큰이었다. 전체 96개 후보 실행은 확인하지 않았다.
주요 오류는 도구 선택 대신 clarify를 반환하는 것, choices를 배열 대신 객체로
출력하는 것, 후보에 없는 스마트폰 카메라 요청을 no_tool로 처리하는 것이었다.
첫 프롬프트는 예시 문자열 복사와 잘못된 action 값도 발생했다. 형식 개선 후에도
정답 일치는 늘지 않았으므로 현재 프롬프트만으로 기존 라우터를 대체할 근거는 없다.

합성 데이터의 미검토 정답과 비교한 소규모 실행 확인이다. 같은 검증 사례를 보며
프롬프트를 수정한 결과는 독립 평가 성능으로 해석하면 안 된다. 이번 실행으로
기존 선형 모델보다 우수하거나 실제 사용에 적합하다고 결론 내릴 수 없다.

실행기 경계 검증은 외부 패키지나 모델 다운로드 없이 실행할 수 있다.

```sh
python3 -m unittest discover -s tests/integration -p test_llm_smoke.py
```
