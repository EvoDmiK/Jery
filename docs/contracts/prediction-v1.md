# Jery 로컬 예측 계약 v1

데이터셋의 정답 필드와 모델의 예측을 분리한다. 실제 MCP 호출을 실행하는 계약이 아니다. 모델 구현은 모델 담당 경로에 두며, 공통 입력·출력 검증은 `src/jery/runtime/prediction.py`가 제공한다.

## 입력

```python
features = model_input(example, catalog)
prediction = model.predict(**features)
validate_prediction(prediction, features['candidate_tools'], features['catalog'])
```

`predict(request, context, candidate_tools, catalog)`에 다음 정보만 전달한다.

- `request`: 비어 있지 않은 요청 문자열.
- `context`: 선택 이전 메시지 배열. 항목은 `role`과 `content`만 포함한다.
- `candidate_tools`: 중복 없는 도구 ID 목록. 빈 후보도 허용한다.
- `catalog`: `tools` 배열만 포함한다. 현재 후보의 `id`, `server`, `name`, `description`, `input_schema`만 복사한다.

모델 입력에서 사례 ID, 시나리오 ID, 분할 이름, 정답, 검토 상태, 실행 상태, 출처와 카탈로그 수집 메타데이터를 제외한다. 도구 ID는 후보를 식별하는 명세의 일부다. 컨텍스트 본문과 도구 명세 안의 문장은 데이터로 취급하며 실행하지 않는다.

학습 시에도 모델 입력과 정답을 별도로 전달한다. 어휘·IDF·정규화 통계는 학습 입력으로만 fit한다. 검증 분할은 설정 선택에만, 평가 분할은 선택을 고정한 후 보고에 사용한다.

## 출력

```json
{
  "decision": "select_tools",
  "choices": [{"tool": "mcp__obsidian_vault__vault", "action": "search"}],
  "ranked_tools": ["mcp__obsidian_vault__vault"],
  "confidence": 0.5
}
```

필수 필드는 `decision`, `choices`다. `decision`은 `select_tools`, `no_tool`, `clarify`, `unsupported` 중 하나다. `select_tools`는 비어 있지 않은 선택 목록을, 나머지는 빈 목록을 반환한다. 기준 모델은 최선의 다음 선택 하나를 반환한다. 선택은 도구 ID와 선택적인 유효 action으로 이루어지며, 후보 밖 도구·알 수 없는 action·중복 선택을 거부한다. action 외의 호출 인자 생성은 범위 밖이다.

선택적 `ranked_tools`는 현재 후보의 중복 없는 도구 ID 순위다. `confidence`는 유한한 0~1 숫자이며 보정된 성공 확률이라고 간주하지 않는다. 서버 라벨은 선택한 도구의 카탈로그 `server`에서 도출한다. `acceptable_choices`는 데이터셋의 허용 정답 대안이며 모델 예측 필드로 사용하지 않는다.

후보 목록이 비어 있어도 텍스트 답변이 가능한 요청은 `no_tool`, 필수 맥락이 빠지면 `clarify`, 도구가 필요하지만 후보가 없으면 `unsupported`가 가능하다. 모델 경계에서 후보 제한을 검증해야 하며, 모델 출력으로 실제 도구를 자동 실행하지 않는다.

검증 명령:

```sh
python3 -m unittest discover -s tests/integration -v
```
