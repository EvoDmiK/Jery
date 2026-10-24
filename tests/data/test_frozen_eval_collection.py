import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('real_eval_collect',ROOT/'scripts/collect_frozen_eval_candidates.py')
collect_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(collect_module)


class FrozenCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'source.db';self.key=self.root/'private/key.json'
        self.catalog=ROOT/'datasets/hermes-ten-tools-v2/tools.json'
        self.db=sqlite3.connect(self.source)
        self.db.executescript('''CREATE TABLE sessions(id TEXT,source TEXT,parent_session_id TEXT,user_id TEXT,display_name TEXT,ended_at INTEGER);
        CREATE TABLE messages(id INTEGER,session_id TEXT,role TEXT,content TEXT,active INTEGER,compacted INTEGER);''')
        self.db.executemany('INSERT INTO sessions VALUES(?,?,?,?,?,?)',[
            ('root','discord',None,'1234567890123456','ExamplePerson',None),
            ('child','discord','root','1234567890123456','ExamplePerson',1),
            ('cron','cron',None,None,None,1)])
        self.db.executemany('INSERT INTO messages VALUES(?,?,?,?,?,?)',[
            (1,'root','user','[ExamplePerson] Swallow 모의계좌의 잔고와 보유목록을 보여줘.',1,0),
            (2,'root','assistant','TRAP_AFTER_CALL obsidian\nBearer sk-secret-value',1,0),
            (3,'root','user','[IMPORTANT: MCP servers have been reloaded. obsidian 30 tools.]',1,0),
            (4,'root','user','[ExamplePerson] Obsidian 이전 기록을 저장해줘.',0,1),
            (5,'root','user','[ExamplePerson] Obsidian 문서를 저장해줘\napi_key=abc-private-key',1,0),
            (6,'child','user','Obsidian 하위 작업자 결과 문서를 저장해줘.',1,0),
            (7,'cron','user','Obsidian 예약 작업 결과 문서를 저장해줘.',1,0),
            (8,'root','user','[ExamplePerson] Obsidian 새 문서 작성 방법을 설명해줘.',1,0)])
        self.db.commit();self.db.close()
        self.prior=self.root/'previous.jsonl'
        self.prior.write_text(json.dumps({'request':'Obsidian 새 문서 작성 방법을 설명해줘.','context':[]},ensure_ascii=False)+'\n')

    def test_target_is_a_goal_not_a_claim_of_collected_or_reviewed_cases(self):
        result = collect_module.collect(self.source, self.catalog, self.root / 'target200', self.key,
                                        [self.prior], target_per_tool=20)
        self.assertEqual(result['target_cases'], 200)
        self.assertEqual(result['target_per_tool'], 20)
        self.assertEqual(result['counts']['candidate_requests'], 1)
        self.assertEqual(result['label_status'], 'unresolved')
        for target in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                collect_module.collect(self.source, self.catalog, self.root / 'bad', self.key, [], target_per_tool=target)

    def collect(self, output):
        return collect_module.collect(self.source,self.catalog,output,self.key,[self.prior])

    def test_read_only_snapshot_filters_system_worker_history_secret_and_prior_duplicates(self):
        before=self.source.read_bytes();out=self.root/'candidates';r=self.collect(out)
        rows=list(map(json.loads,(out/'cases.jsonl').read_text().splitlines()))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['request'],'Swallow 모의계좌의 잔고와 보유목록을 보여줘.')
        self.assertEqual(rows[0]['context'],[])
        self.assertIsNone(rows[0]['decision']);self.assertEqual(rows[0]['acceptable_choices'],[])
        self.assertEqual(rows[0]['review_status'],'needs_human_review');self.assertFalse(rows[0]['execution_verified'])
        self.assertEqual(r['status'],'candidate')
        self.assertEqual(r['counts']['injected_non_user_notice'],1)
        self.assertEqual(r['counts']['credential_bearing_request'],1)
        self.assertEqual(r['counts']['previously_observed_request'],1)
        self.assertEqual(before,self.source.read_bytes())
        raw=(out/'cases.jsonl').read_text()
        for forbidden in ('ExamplePerson','TRAP_AFTER_CALL','abc-private-key','1234567890123456'):
            self.assertNotIn(forbidden,raw)
        self.assertEqual((out/'cases.jsonl').stat().st_mode&0o777,0o600)
        self.assertEqual(self.key.stat().st_mode&0o777,0o600)
        repeat=self.root/'repeat';self.collect(repeat)
        self.assertEqual((out/'cases.jsonl').read_bytes(),(repeat/'cases.jsonl').read_bytes())

    def test_rejects_overwrite_wrong_source_and_key_aliases(self):
        out=self.root/'candidates';self.collect(out)
        with self.assertRaises(ValueError):self.collect(out)
        other=self.root/'other.db';other.write_bytes(self.source.read_bytes())
        with self.assertRaises(ValueError):collect_module.collect(other,self.catalog,self.root/'new',self.key,[])
        with self.assertRaises(ValueError):collect_module.collect(self.source,self.catalog,self.root/'new',self.source,[])
        missing=self.root/'missing.db'
        with self.assertRaises(ValueError):collect_module.collect(missing,self.catalog,self.root/'new',missing,[])
        self.assertFalse(missing.exists())

    def test_names_are_case_insensitive_short_names_redacted_and_public_brackets_preserved(self):
        db=sqlite3.connect(self.source)
        db.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)',('kor','discord',None,'user_k','이예시',None))
        db.executemany('INSERT INTO messages VALUES(?,?,?,?,?,?)',[
            (9,'root','user','[exampleperson] Swallow 전체 잔고를 ExamplePerson에게 보여줘.',1,0),
            (10,'kor','user','[이예시] Swallow 모의계좌 자산을 이예시님에게 보여줘.',1,0),
            (11,'root','user','[Obsidian] 새 메모 파일 목록을 확인해줘.',1,0)])
        db.commit();db.close()
        out=self.root/'names';self.collect(out)
        rows=list(map(json.loads,(out/'cases.jsonl').read_text().splitlines()))
        text=(out/'cases.jsonl').read_text().casefold()
        self.assertNotIn('exampleperson',text);self.assertNotIn('이예시',text)
        self.assertTrue(any(r['request'].startswith('[Obsidian]') for r in rows))
        self.assertEqual(collect_module.normalized('문서 [path_aaaaaaaa] 읽어줘'),
                         collect_module.normalized('문서 [path_bbbbbbbb] 읽어줘'))


if __name__=='__main__':unittest.main()
