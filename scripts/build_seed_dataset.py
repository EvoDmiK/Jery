"""Build deterministic synthetic routing examples from a saved MCP catalog."""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/hermes-routing-v1'

# Each line is one independent scenario with two Korean paraphrases.
# Routing is the next decision, not a full plan or executable call.
SEEDS = {
    'obsidian_vault.vault': '''search|볼트에서 MCP 관련 노트를 찾아줘.|내 Obsidian 안에서 MCP를 다룬 문서 검색해줘.
list|Obsidian Projects 폴더에 어떤 파일이 있는지 보여줘.|볼트의 Projects 디렉터리 파일 목록을 조회해줘.
create|볼트에 Ideas/Jery.md를 만들고 내용은 '도구 선택 모델'로 저장해줘.|Obsidian Ideas/Jery.md 파일을 새로 생성해서 '도구 선택 모델'이라고 적어줘.
rename|볼트의 Draft.md 파일명을 Final.md로 바꿔줘.|Obsidian에서 Draft.md를 Final.md라는 이름으로 변경해줘.
search|볼트에서 #robotics 태그가 붙은 문서를 찾아줘.|내 Obsidian 노트 중 robotics 태그를 가진 것들을 검색해줘.''',
    'obsidian_vault.edit': '''append|볼트의 Jery.md 끝에 '다음 단계: 로그 수집'을 추가해줘.|Obsidian Jery.md 맨 아래에 '다음 단계: 로그 수집'이라는 줄을 붙여줘.
window|볼트의 Plan.md에서 '초안'을 '검토 중'으로 바꿔줘.|Obsidian Plan.md 안의 '초안' 문구를 찾아 '검토 중'으로 치환해줘.
at_line|볼트의 Notes.md 10번째 줄 앞에 '확인 필요'를 삽입해줘.|Obsidian Notes.md의 10번 줄 바로 위에 '확인 필요'라는 줄을 넣어줘.
patch|볼트의 Project.md frontmatter에서 status 값을 done으로 수정해줘.|Obsidian Project.md의 frontmatter status 필드를 done으로 변경해줘.
patch|볼트의 Roadmap.md에서 '할 일' 제목 아래에 '데이터 검토' 항목을 추가해줘.|Obsidian Roadmap.md의 할 일 섹션에 '데이터 검토'를 덧붙여줘.''',
    'obsidian_vault.view': '''active|Obsidian에서 지금 편집 중인 노트를 보여줘.|현재 Obsidian 에디터에 열린 문서 내용을 조회해줘.
window|볼트의 Jery.md 40번째 줄 주변을 보여줘.|Obsidian Jery.md에서 40번 줄을 중심으로 앞뒤 내용을 보고 싶어.
open_in_obsidian|볼트의 Projects/Jery.md를 Obsidian 앱에서 열어줘.|Obsidian 앱으로 Projects/Jery.md 노트를 열어줘.
window|볼트의 Notes.md에서 '평가 기준'이라는 문구 주변만 보여줘.|Obsidian Notes.md 안에서 '평가 기준' 근처 문맥을 조회해줘.
active|Obsidian의 현재 활성 문서 내용을 확인해줘.|지금 선택된 Obsidian 노트의 본문을 가져와줘.''',
    'obsidian_vault.graph': '''backlinks|볼트의 Jery.md를 링크하는 다른 노트들을 찾아줘.|Obsidian Jery.md로 들어오는 백링크 목록을 보여줘.
forwardlinks|볼트의 MCP.md에서 나가는 문서 링크를 보여줘.|Obsidian MCP.md가 링크하는 노트 목록을 조회해줘.
path|볼트에서 Jery.md와 Hermes.md가 링크로 어떻게 이어지는지 찾아줘.|Obsidian Jery.md부터 Hermes.md까지 연결 경로를 확인해줘.
traverse|볼트의 Routing.md에서 링크를 따라 두 단계까지 연결된 노트를 탐색해줘.|Obsidian Routing.md를 출발점으로 링크 그래프를 깊이 2까지 순회해줘.
statistics|Obsidian 볼트 전체의 링크 연결 통계를 보여줘.|내 볼트의 문서 링크 수와 연결 구조 통계를 조회해줘.''',
    'swallow_trader.swallow_get_stock_price': '''-|삼성전자 현재 주가를 조회해줘.|지금 삼성전자 주식 가격이 얼마인지 가져와줘.
-|005930의 현재가와 전일 대비 등락률을 보여줘.|종목코드 005930의 지금 가격과 등락률을 확인해줘.
-|현대차 주식의 현재 가격을 알려줘.|현대차 실시간 주가를 조회해줘.
-|SK하이닉스 지금 주가가 얼마야?|SK하이닉스의 현재 시세를 가져와줘.
-|카카오 현재 주가와 장 운영 상태를 조회해줘.|지금 카카오 시세와 거래장 상태를 확인해줘.''',
    'swallow_trader.swallow_get_stock_news': '''-|삼성전자 관련 최신 주식 뉴스를 찾아줘.|삼성전자에 관한 최근 증권 기사를 검색해줘.
-|반도체 업종 뉴스 다섯 개를 가져와줘.|반도체 산업 관련 최신 주식 기사 5개를 조회해줘.
-|오늘 주요 증권 헤드라인을 보여줘.|특정 종목 지정 없이 최신 주식 뉴스 목록을 가져와줘.
-|현대차 관련 주식 뉴스 열 개를 검색해줘.|현대차에 대한 최신 증권 기사 10건을 찾아줘.
-|배터리 산업 관련 주식 뉴스를 조회해줘.|최근 배터리 업종에 관한 증권 뉴스를 검색해줘.''',
    'swallow_trader.swallow_get_price_history': '''-|삼성전자의 지난 30일 종가 이력을 조회해줘.|삼성전자 과거 30일 일별 종가를 가져와줘.
-|005930의 토스와 네이버 종가를 비교해줘.|종목 005930의 과거 종가를 토스와 네이버 소스로 교차검증해줘.
-|현대차의 지난 90일 종가를 보여줘.|현대차 주식의 최근 90일 가격 이력을 조회해줘.
-|SK하이닉스 지난 7일 종가를 네이버 기준으로 가져와줘.|네이버 소스에서 SK하이닉스의 최근 7일 종가 이력을 확인해줘.
-|카카오의 과거 60일 종가를 토스 기준으로 조회해줘.|토스 소스로 카카오의 지난 60일 가격 기록을 가져와줘.''',
    'swallow_trader.swallow_get_portfolio': '''-|Swallow 모의투자 계좌의 총 자산을 조회해줘.|Swallow 가상 투자 계좌가 현재 얼마인지 확인해줘.
-|Swallow 모의계좌 예수금 잔고를 보여줘.|내 Swallow 가상 계좌에 남은 현금을 조회해줘.
-|Swallow 모의투자 보유 종목 목록을 보여줘.|Swallow 가상 포트폴리오에 어떤 주식을 갖고 있는지 확인해줘.
-|Swallow 모의투자 전체 평가손익을 조회해줘.|내 Swallow 가상 포트폴리오의 현재 평가 손익을 가져와줘.
-|Swallow 모의계좌 보유 종목들의 평균단가를 보여줘.|Swallow 가상 계좌에 있는 주식들의 평균 매입 가격을 조회해줘.''',
    'swallow_trader.swallow_get_trades': '''-|Swallow 모의투자 최근 체결 내역을 보여줘.|Swallow 가상 계좌의 최근 매매 기록을 조회해줘.
-|Swallow에서 005930 매매 이력만 가져와줘.|Swallow 모의투자 종목코드 005930의 체결 기록을 확인해줘.
-|Swallow 최근 거래 20건을 조회해줘.|Swallow 가상 매매 체결 내역을 최신 20개 보여줘.
-|Swallow 모의투자 최근 매매의 실현 손익을 확인해줘.|Swallow에서 최근 체결된 거래별 실현 손익 기록을 가져와줘.
-|Swallow 최근 체결 기록에 남긴 AI 시그널 근거를 보여줘.|Swallow 가상 매매 이력에서 주문 근거로 기록된 AI 시그널을 조회해줘.''',
    'kestrel.get_codex_reset_status': '''-|Codex 사용량 리셋 현황을 조회해줘.|Codex 사용량 초기화 상태를 확인해줘.
-|Codex 최근 리셋 이벤트가 언제였는지 조회해줘.|Codex 사용량이 최근에 리셋된 이벤트 시각을 확인해줘.
-|Codex 예정된 리셋 이벤트가 있는지 찾아줘.|Codex 사용량 초기화 예정 이벤트를 조회해줘.
-|Codex 리셋 서비스에서 현재 리셋 상태를 가져와줘.|Codex 사용량 리셋 상태 서비스의 최신 정보를 조회해줘.
-|Codex 리셋 이벤트 시각을 조회해서 한국 시간으로 알려줘.|Codex 사용량 리셋 이벤트 시간을 가져와 KST로 표시해줘.''',
}
SPECIAL = {
    'no_tool': '''안녕, 반가워!|안녕하세요!
MCP라는 약어를 풀어서 설명해줘.|Model Context Protocol이 무엇인지 일반적으로 설명해줘.
이 문장을 영어로 번역해줘: 나는 데이터를 모으고 있다.|'나는 데이터를 모으고 있다'를 영어 문장으로 바꿔줘.
다음 글을 요약해줘: Jery는 로컬에서 도구를 선택하는 모델이다.|'Jery는 로컬에서 도구를 선택하는 모델이다'라는 문장을 짧게 요약해줘.
과거 가격 이력과 현재가의 차이를 개념적으로 설명해줘.|현재 주가와 과거 종가 데이터가 어떻게 다른지 설명해줘.''',
    'clarify': '''그 종목 현재가 조회해줘.|아까 그 주식 지금 가격 알려줘.
그 노트의 이름을 바꿔줘.|그 문서를 다른 이름으로 변경해줘.
그 파일에서 문구를 바꿔줘.|그 노트 내용을 수정해줘.
두 노트 사이의 링크 경로를 찾아줘.|그 문서 둘이 링크로 어떻게 연결되는지 확인해줘.
그 주식의 가격 이력을 가져와줘.|그 종목 과거 종가를 조회해줘.''',
    'unsupported': '''내 Google 캘린더에 내일 오후 3시 회의를 등록해줘.|내일 15시에 Google Calendar 일정으로 회의를 생성해줘.
서울 내일 날씨를 기상 서비스에서 조회해줘.|기상 API로 내일 서울의 날씨 예보를 가져와줘.
Swallow 모의계좌를 초기화해줘.|Swallow 가상 포트폴리오를 리셋해줘.
내 이메일 받은편지함을 검색해줘.|메일 계정의 수신 메시지 목록을 조회해줘.
Swallow 모의계좌에서 005930 삼성전자를 단가 80000원으로 가상 매수해줘.|Swallow에서 종목코드 005930, 종목명 삼성전자, 주문단가 80000원으로 가상 BUY 주문을 넣어줘.''',
}


def build():
    catalog = json.loads((OUT / 'tools.json').read_text())
    by_key = {t['server'] + '.' + t['name']: t['id'] for t in catalog['tools']}
    version = catalog['catalog_version']
    rows = []
    for label, block in {**SEEDS, **SPECIAL}.items():
        for i, line in enumerate(block.splitlines()):
            split = 'train' if i < 3 else 'validation' if i == 3 else 'test'
            parts = line.split('|')
            action, prompts = (parts[0], parts[1:]) if label in SEEDS else (None, parts)
            for j, prompt in enumerate(prompts):
                uid = f'{label}.{i + 1}.{j + 1}'
                candidates = sorted(by_key.values())
                random.Random(uid).shuffle(candidates)
                choice = [] if label not in SEEDS else [{'tool': by_key[label], **({'action': action} if action != '-' else {})}]
                rows.append({
                    'id': uid, 'scenario_id': f'{label}.{i + 1}', 'split': split,
                    'request': prompt, 'context': [], 'candidate_tools': candidates,
                    'tool_catalog_version': version,
                    'decision': 'select_tools' if label in SEEDS else label,
                    'acceptable_choices': choice, 'source': 'synthetic',
                    'review_status': 'needs_human_review',
                    'execution_verified': False,
                })
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'validation', 'test'):
        selected = [r for r in rows if r['split'] == split]
        random.Random('jery-v1-' + split).shuffle(selected)
        (OUT / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in selected))
    files = ['tools.json', 'train.jsonl', 'validation.jsonl', 'test.jsonl']
    manifest = {
        'dataset_version': 'hermes-routing-v1', 'catalog_version': version,
        'task': 'next_step_tool_routing', 'language': 'ko',
        'synthetic': True, 'human_reviewed': False,
        'tools': len(catalog['tools']), 'examples': len(rows),
        'split_counts': dict(Counter(r['split'] for r in rows)),
        'decision_counts': dict(Counter(r['decision'] for r in rows)),
        'sha256': {f: hashlib.sha256((OUT / f).read_bytes()).hexdigest() for f in files},
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    build()
