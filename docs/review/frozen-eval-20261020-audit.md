# 독립 평가셋(`frozen-eval-v1` / `candidates-v4`) 누수·출처·정답 검토 감사 보고서

## 1. 감사 대상 및 범위

루트(`dev-codex`)가 실제 Hermes 상태 DB(`hermes-profile-1`, 스냅샷 SHA-256 `26f85bcb418e88f7b1d86be612d04af25ba5f0b4022b9d45a730297902da21a5`)에서 준식별(quasi-identifying) 서술 1건을 추가 격리(`privacy_quarantine: 1`)한 뒤 내보낸 정식 후보 패킷 `data/local/frozen-eval-20261020/candidates-v4/`(13건)와 Antigravity 독립 라벨 제안서(`outputs/frozen-eval-20261020/agy-proposals.json`)를 `apps/review/frozen_eval_audit.py`로 읽기 전용 검증했다. 이전 `candidates`, `candidates-v2`, `candidates-v3`는 대체된 비공개 초안으로 동결 대상이 아니다.

- **개인정보·기밀 보호 준수**: 13건의 실제 사용자 요청 원문, 닉네임, 개인 식별자는 본 문서·터미널·Git 커밋에 일절 인용하지 않고 사례 ID, 해시, 집계 통계로만 보고한다.
- **대상 파일 및 SHA-256**:
  - `data/local/frozen-eval-20261020/candidates-v4/cases.jsonl`: `b4d65338621c87a98600658a77e21514dbd03ec0a9ed8b840db7d7408e2ccdff` (13행, 10개 고유 세션 그룹)
  - `data/local/frozen-eval-20261020/candidates-v4/candidate-manifest.json`: `8df872d79d191f775e23591951cf0afffb892a8e2fc13da62fe63dc4a1698bd5` (`schema_version=frozen-eval-candidates-v1`, `status=candidate`, 수집기 코드 SHA-256 `83ae29bb1b76bf3f418a2f88273753f142d45805207c254ddb2bfdb88ecc929a`)
  - `datasets/hermes-ten-tools-v2/tools.json`: `11c0253a5de892afaec5836b847cfb70bc76c4caf23e96b8b41a414183123847` (`catalog_version=hermes-ten-tools-2289142ab57a`, 10개 고정 도구)
  - `outputs/frozen-eval-20261020/agy-proposals.json`: `4c6fdf65f2470919d484fffb9966419ca57ec4235df1ca5a3c1f090dce8fc8b2` (13개 사례별 Antigravity 독립 제안)
  - `outputs/frozen-eval-20261020/agy-audit-v4.json`: 결정론적 감사 결과 JSON (Git 무시 경로)

---

## 2. 과거 코퍼스(1,484건) 대비 누수·출처·개인정보 위생 감사 결과

기존 8개 데이터 파일(`hermes-routing-v1` train/val/test 130건, `mcp-routing-v2` train/val/test 1,162건, `hermes-ten-tools-contrast-source-v1/cases.jsonl` 180건, `diagnostics/laya-ko-en-request-v1.json` 12건 — 총 **1,484건**, 코퍼스 지문 `5f7f245963f129f15142d333fa6ad5611b3a5c3218898aa74ed61bb28c28afc4`) 전체와 대조한 결과는 다음과 같다(정규화 서명 비교 시 `[url_<hmac>]` 등 가명 플레이스홀더의 HMAC 값은 무시하고 `[url]` 등으로 정규화해 대조).

| 검증 항목 | 결과 | 세부 수치 |
|---|---|---|
| 패킷 내부 ID / 정규화 요청 / 상태 중복 | **PASS** | 중복 `0`건 (`unique_ids=13`, `unique_scenarios=10`) |
| 과거 코퍼스 ID / `scenario_id` 충돌 | **PASS** | 충돌 `0`건 |
| 과거 코퍼스 정규화 요청 / 상태 일치 | **PASS** | 일치 `0`건 (`zero_exact_leakage=true`) |
| 과거 코퍼스 근접 중복 (3-gram+단어 Jaccard `>= 0.55`) | **PASS** | 근접 중복 `0`건 |
| 출처 진위 (`source.kind`) 및 발화 주체 (`actor_kind`) | **PASS** | 13건 모두 `kind="real"`, `actor_kind="user"` (`human_origin_verified=false`, `standalone_label_review_required=true` 13건, `context_dependency_suspected=true` 10건), 합성 `0`건, 자동화/크론/워커 공지 `0`건 |
| 대괄호 화자 닉네임 / 미정제 PII / 호출 후 결과 누수 | **PASS** | 화자 닉네임 잔존 `0`건, 준식별 이력 서술 격리 완료(`privacy_quarantine=1`), 미정제 이메일·경로·토큰 `0`건, 호출 후 필드 및 도구 ID 노출 `0`건 |
| 패킷·검토·평가 게이트 상태 분리 | **PASS** | `packet_status="candidate"`, `label_review_gate="unresolved_gold_with_ai_proposals_needs_human_review"`, `evaluation_gate="refused"`, `freeze_implies_human_verified=false` |

---

## 3. Antigravity 독립 라벨 검토(`agy-proposals.json`, 13건) 집계 및 사유 분류

모델 예측이나 타 검토자(Root, Claude)의 라벨을 일절 참조하지 않고, 고정 10개 도구 카탈로그(`hermes-ten-tools-2289142ab57a`) 명세와 단일 턴 입력(`request`, `context=[]`) 완결성만 기준으로 13건을 독립 분류했다.

### 3.1 Disposition 및 Decision 분포 요약
- **`label` (`decision="select_tools"`, 단일 도구 직접 실행 가능): 1건 (`confidence="high"`)**
  - `real_eval_e3ae9faea70b43fb03e224a009960702` → `mcp__swallow_trader__swallow_get_portfolio` (`action: null`): 사전 맥락 없이도 모의투자 계좌 포트폴리오 및 잔고 현황 조회가 카탈로그 단일 도구로 완결된다.
- **`needs_context` (`decision="clarify"`, 이전 대화·문서 본문 누락): 5건 (`confidence="high"`)**
  - `real_eval_0539063f4d2ca51b55978bead4f07edd`, `real_eval_7b4eb4f9e38c72f7127a0cb06098012c`, `real_eval_8a9771b39825590f20efa242b7b66226`, `real_eval_b85361053a27891cff343aca9d2cfb93`, `real_eval_d6ed5e72b63e140beb0f5bc0fa661b6f`
  - 단독 사용자 메시지 스냅샷(`context=[]`) 상태에서 이전 턴에 생성·변환·언급된 문서 내용이나 대상 노트 식별자를 지시어/후속 질문으로만 참조하고 있어, 사전 문맥 없이는 단일 도구 호출 인자를 구성할 수 없다(수집기 `context_dependency_suspected=true`와 일치).
- **`unsupported` (`decision="unsupported"`, 고정 10개 도구 밖 기능 요청): 2건 (`confidence="high"`)**
  - `real_eval_dd4c1e983151f2498d1458f801332373`: 관심종목 추가 및 지속 모니터링 요청(`mcp__swallow_trader__swallow_watchlist` 등 고정 10개 도구 카탈로그 밖 기능).
  - `real_eval_e7c474f575abc113036ad5f27230026b`: 모의투자 포트폴리오 초기화 요청(`mcp__swallow_trader__swallow_reset_portfolio`는 고정 10개 도구 카탈로그에 미포함).
- **`exclude` (일반 조언·향후 규칙 선언 `no_tool` 2건 + 외부 스킬 결합 다단계 워크플로 `unsupported` 3건): 5건 (`confidence="high"`)**
  - `real_eval_8b98c619f13ff600645383058c4d7feb` (`decision="no_tool"`): NAS 양방향 백업 구성을 묻는 일반 기술 자문 질문으로 즉시 실행할 MCP 도구 호출 대상이 아니다.
  - `real_eval_cde28ddc5719f2cd8a45944b3eb55151` (`decision="no_tool"`): 향후 검증 기록 작성 방식에 대한 메타 규칙 지시문으로 현재 턴에 저장할 본문이 없다.
  - `real_eval_80e0bcd98b8810b2a32ab8176760afdb`, `real_eval_eaabe5db368676cf2be442abbe0af6ab`, `real_eval_d26cd9cf30efdbe06e489be5f2ca0f75` (`decision="unsupported"`): 외부 URL 문서 수집 및 카탈로그 밖 `markdown-converter` 스킬 변환을 선행해야 하는 다단계 워크플로이거나 이전 턴 변환 작업에 의존하는 연속 지시다.

---

## 4. 고정 10개 도구별 적격 사례 수 및 목표(100건) 대비 부족분 정량화

루트 DB의 전체 루트 Discord 사용자 메시지 2,050건 중 길이 제외(55건), 도구 무관(381건), 비활성/압축(1,581건), 시스템 주입 공지(19건), 개인정보 준식별 격리(1건)를 제외한 실제 후보가 13건이며, 그중 단일 10개 도구 선택 과제(`ten_tool_choice`)에 바로 사용 가능한 적격(`disposition="label"`, `decision="select_tools"`) 사례는 **1건**(총 부족분 **99/100건**), 인간 검토 완료(`human_reviewed`) 사례는 **0건**(총 부족분 **100/100건**)이다. 따라서 현재 소스만으로는 100건의 독립 실제 요청 벤치마크를 동결·완성할 수 없으며, `evaluation_gate`는 `"refused"` 상태를 유지한다.

| 고정 10개 도구 ID | 목표 수 | 인간 검토 완료 (`human_reviewed`) | AI 제안 적격 (`select_tools`) | 목표 대비 부족분 (AI 적격 기준) | 목표 대비 부족분 (인간 검토 기준) |
|---|---:|---:|---:|---:|---:|
| `mcp__obsidian_vault__vault` | 10 | 0 | 0 | 10 | 10 |
| `mcp__obsidian_vault__edit` | 10 | 0 | 0 | 10 | 10 |
| `mcp__obsidian_vault__graph` | 10 | 0 | 0 | 10 | 10 |
| `mcp__obsidian_vault__dataview` | 10 | 0 | 0 | 10 | 10 |
| `mcp__obsidian_vault__bases` | 10 | 0 | 0 | 10 | 10 |
| `mcp__swallow_trader__swallow_get_stock_price` | 10 | 0 | 0 | 10 | 10 |
| `mcp__swallow_trader__swallow_get_price_history` | 10 | 0 | 0 | 10 | 10 |
| `mcp__swallow_trader__swallow_get_portfolio` | 10 | 0 | 1 | 9 | 10 |
| `mcp__swallow_trader__swallow_order` | 10 | 0 | 0 | 10 | 10 |
| `mcp__swallow_trader__swallow_backtest` | 10 | 0 | 0 | 10 | 10 |
| **합계** | **100** | **0** | **1** | **99** | **100** |

---

## 5. 재현 및 검증 명령

```sh
# 1. 실제 candidates-v4 (13건) 및 agy-proposals.json 대상 결정론적 감사 실행
python3 apps/review/frozen_eval_audit.py \
  --packet-dir /Volumes/Dove-Nest-SSD/projects/Jery/data/local/frozen-eval-20261020/candidates-v4 \
  --proposals /Volumes/Dove-Nest-SSD/projects/Jery/outputs/frozen-eval-20261020/agy-proposals.json \
  --output-json /Volumes/Dove-Nest-SSD/projects/Jery/outputs/frozen-eval-20261020/agy-audit-v4.json \
  --json

# 2. 리뷰 감사기 단위 테스트 스위트 실행
python3 -m unittest discover -s tests/review -v
```
