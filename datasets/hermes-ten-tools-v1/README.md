# Hermes 10개 도구 선택 진단 데이터 v1

등록 카탈로그 `hermes-mcp-57e701d87790`에서 Obsidian 5개·Swallow 5개 도구를 선택했다. 원본 `mcp-routing-v2`를 덮어쓰지 않는다. 고정 후보 10개를 한 번에 제시하는 `select_tools` 전용 과제이며, null 판단과 action 선택은 포함하지 않는다. 실제 도구를 호출하지 않는다.

원본 request+context 중복을 제거하고 시나리오 분할 경계를 유지했다. memorization profile은 train 20·validation 20·test 20개, full profile은 train 60·validation 20·test 20개다. 각 클래스의 행 수가 같다. 두 profile의 validation/test는 선택지 순서까지 동일하다. metadata의 원 후보·원 정답을 보존하지만 학습 특징으로 사용하지 않는다.

선택 기준은 `selection.json`, 10개 실제 도구의 명세는 `tools.json`, 원본·카탈로그·profile의 해시는 `manifest.json`에 있다. 모델 입력은 Git 밖 `outputs/laya-ten-tools-20261020/{memorization,full}`의 `decision-export-v2` JSONL이다. MLflow 새 Parent에 데이터와 manifest를 JSON artifact로 저장한다.

원래 합성 정답은 미검토 상태다. 이 진단을 실사용 정확도로 해석하지 않는다. test는 이미 관찰된 원본의 회귀 평가이며 독립적인 신규 평가셋이 아니다. 원본 후보 집합을 고정된 10개로 바꾸었으므로 기존 96개 도구 과제의 정확도와 직접 비교하지 않는다.

재현 시 새 경로를 지정한다.

```sh
python3 scripts/build_laya_ten_tools.py \
  --dataset-dir outputs/ten-tools-rebuild/catalog \
  --output outputs/ten-tools-rebuild/export
python3 -m unittest tests.data.test_laya_ten_tools
```

상세 계약은 [decision-export-v2](../../docs/contracts/decision-export-v2.md)를 따른다. 기존 v1 파일은 변경하지 않는다.
