# 10개 도구 오류 검토와 v2 데이터 보강

10개 배치·시드 실험의 **선정 head validation 예측**만 결합했다. Test 예측은 오류 분석에 사용하지 않았다. 실행별 20개 source_id가 정확히 한 번씩 있는지, 10개 run_id가 고유한지, 후보와 정답 참조가 올바른지 검사했다. 200개 예측 중 80개 오류, 20개 사례 중 10개에서 적어도 한 번 오류, 4개는 10회 모두 오류였다. 반복 예측은 독립 사례 200개가 아니라 같은 검증 20개를 반복한 것이다.

| 반복 오류 | 오류 실행 수 | 카탈로그 대조 |
|---|---:|---|
| Bases 뷰 → Dataview | 10/10 | bases.view와 viewName 지원, 기존 짧은 설명은 뷰 조회 생략 |
| 종가 이력 → 현재가 | 10/10 | history의 days/source 지원 |
| 한 종목 보유 여부 → 다른 도구 | 10/10 | stock_price 설명에 개별 보유 현황 포함 |
| 계좌 전체 평가손익 → 주문/가격 이력 | 10/10 | portfolio 설명에 전체 평가손익 포함 |
| frontmatter 수정 2건 → 다른 도구 | 9/10, 7/10 | edit.patch의 frontmatter 지원, 짧은 설명 생략 |
| 파일 이름 변경 → 다른 도구 | 4/10 | vault.rename 지원, 짧은 설명 생략 |

근거가 확인된 문제는 일부 기능 설명의 불완전성이다(높은 확신). 이 문제가 실제 오류의 원인이라는 인과 결론은 아직 없다. 기존 라벨과 카탈로그의 직접 충돌은 확인하지 못했지만, 카탈로그만으로 유일한 최적 선택이나 실행 성공을 보장할 수 없다. AI 대조 결과는 `datasets/hermes-ten-tools-v2/label-inspection.json`이며 human review와 execution 상태는 승격하지 않았다.

180개 신규 합성 학습 사례를 60개 시나리오로 작성했다. 각 시나리오의 직접 요청·표현 변형·맥락 해결형 3개를 모두 train에 둔다. 기존 60개와 합치면 240개, 도구별 24개다. 기능 설명은 누락된 rename·frontmatter·Bases view와 종목/계좌·현재/과거 가격의 범위를 보완했다. 새 데이터 버전은 hermes-ten-tools-v2, export 계약은 decision-export-v2다.

누수 방지를 위해 정규화된 입력 중복과 scenario 교집합을 거부한다. 초기 초안에서 기존 validation과 같은 Dataview 상태 문장 1건이 차단돼 해당 family 전체를 폐기하고 별도 CALENDAR 질의 family로 교체했다. 근접 의미까지 자동으로 독립성을 보증하는 것은 아니다. 검증 오류에서 얻은 기능 경계로 작성했으므로 validation-informed 합성 데이터라는 한계를 유지한다.

두 profile은 설명만 바꾼 대조군(train 60)과 학습 보강군(train 240)이다. Validation/test 각 20개는 두 profile에서 동일하며 원본의 state·gold·aliases·scenario·순서를 보존했다. 설명과 catalog provenance는 명시적으로 바뀌었으므로 이전 입력과 동일하지 않다. 기존 v1 데이터·export의 해시는 유지된다.

성능 확인용 실행은 같은 seed3407, batch4, LR0.0003538647078409709, weight_decay0.05, smoothing0.05, warmup10, 같은 base를 쓴다. 대조군 32에폭×15step, 보강군 8에폭×60step으로 최대 optimizer 예산을 480step에 맞췄다. 조기 종료와 평가 주기(15 vs 60step)는 다르므로 단일 시드의 회귀 실험이며 개별 인과 효과나 실사용 개선의 증거로 해석하지 않는다. Test를 튜닝에 사용하지 않는다.

재현:

```sh
python3 scripts/audit_ten_tool_errors.py \
  --validation outputs/laya-ten-tools-20261020/full/validation.jsonl \
  --predictions outputs/laya-ten-tools-error-review-20261020/validation-error-input.json \
  --output outputs/laya-ten-tools-error-review-20261020/error-audit-new.json
python3 scripts/build_ten_tool_contrast.py \
  --dataset /tmp/jery-ten-tools-v2-new \
  --output /tmp/jery-ten-tools-v2-exports-new
PYTHONPATH=src python3 -m unittest discover -s tests/data -p test_ten_tool_contrast.py
```

출처 prediction input과 감사 JSON은 outputs에 보존하고 MLflow artifact로 등록한다. 원본 예측 파일은 batch-seed Parent 아래 Child별 details 결과에서도 복원할 수 있다. 동반 notebook은 같은 감사 함수를 사용하며 입력 export·예측 JSON이 필요하다. 실제 로그 수집·사람 정답 검토·신규 독립 평가가 남아 있다.

검증: 학습용 Python에서 전체 151개 테스트 통과(44.998초), 기본 v1 데이터 검증 통과. 동반 notebook의 코드 셀 4개는 Python으로 순서 실행해 출력을 저장했다. Jupyter kernel 검증은 nbformat/nbclient/ipykernel 미설치로 실행하지 않았으며 notebook에 별도 재현 명령과 한계를 적었다. 새 export는 회사 서버에서 native builder/tokenizer 검사 후 실행한다.

Native 사전검사에서 초안 설명의 일부가 head의 옵션당 24token 한도를 넘어 6개 옵션이 잘리는 것을 발견해 학습을 차단했다. 초기 초안은 `outputs/laya-ten-tools-contrast-20261020-native-draft/`로 보존하고, 핵심 기능을 유지한 짧은 설명으로 representation_version을 v2.1-native-budget으로 바꿨다. Catalog version도 명세 해시에 따라 바뀐다. v1 원본과 라벨 의미는 그대로이며 새 설명의 native 재검사를 통과하기 전 학습하지 않는다.

최종 catalog `hermes-ten-tools-2289142ab57a`의 native 재검사에서 두 profile 모두 skip·target 매핑 불일치·특징 불일치·잘림 0건을 확인했다(최대 전체 길이 289token). 첫 draft의 설명만 native 검사에서 실패했고 모델 학습은 시작하지 않았다.

회사 서버에서 보강 Parent `5f64008a2b974b7f87fab5088bafcf8a` / Child `5ee0cebebfbd465898abdd4b5f300a56`, 설명만 보완 Parent `f7faeae2708945d6b62f59e54a6f56ae` / Child `6145fe8c826a4fcea9f23fc525535aec`를 실행했다. 보강군 epoch3·180step, 대조군 epoch9·135step까지 유한 loss/gradient와 CUDA 실행을 확인했다. 감사 input·결과·라벨 대조·criteria·합성 원문·notebook·native readiness를 각 Parent의 data-quality/readiness artifact에 업로드하고 SHA-256 read-back으로 검증한다.
