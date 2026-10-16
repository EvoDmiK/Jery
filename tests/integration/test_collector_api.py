import importlib.util
import json
import shutil
import subprocess
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location('collector_fixture', Path(__file__).parents[1] / 'data/test_collector.py')
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)
from jery.runtime.collector_api import make_server


class APITest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture.CollectorTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.server = make_server(self.fixture.collector, 'test-token-long-enough', port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.close)
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self, path, data=None, token='test-token-long-enough'):
        headers = {'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}
        req = Request(self.url + path, headers=headers, data=json.dumps(data).encode() if data is not None else None)
        try:
            with urlopen(req, timeout=5) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_auth_validation_and_n8n_roundtrip(self):
        self.assertEqual(self.request('/v1/routing-candidates', token='wrong')[0], 401)
        self.assertEqual(self.request('/v1/routing-candidates?limit=0')[0], 400)
        self.assertEqual(self.request('/v1/routing-candidates?consumer=a&consumer=b')[0], 400)
        self.assertEqual(self.request('/unknown')[0], 404)
        status, batch = self.request('/v1/routing-candidates')
        self.assertEqual(status, 200)
        self.assertIsNotNone(shutil.which('node'), 'Node is needed to check the existing n8n JavaScript')
        workflow = json.loads((Path(__file__).parents[2] / 'workflows/n8n/hermes-logs-to-review.json').read_text())
        validate = next(n['parameters']['jsCode'] for n in workflow['nodes'] if n['id'] == 'validate')
        program = '''const fs = require('fs'); const batch=JSON.parse(fs.readFileSync(0,'utf8'));
const $input={first:()=>({json:batch})};
const $=()=>({first:()=>({json:{consumer:'jery-dataset-v1',limit:100}})});
const run = new Function('$input','$',PROCESS_CODE);
console.log(JSON.stringify(run($input,$)[0].json));'''.replace('PROCESS_CODE', json.dumps(validate))
        result = subprocess.run(['node', '-e', program], input=json.dumps(batch), capture_output=True, text=True, check=True)
        submission = json.loads(result.stdout)
        self.assertEqual(submission, self.fixture.collector.submission(batch))
        status, receipt = self.request('/v1/review-batches', submission)
        self.assertEqual(status, 200)
        self.assertTrue(receipt['committed'])
        self.assertEqual(self.request('/v1/review-batches', submission), (status, receipt))
        submission['examples'][0]['observed_choices'][0]['tool'] = 'mcp__unknown__tool'
        self.assertEqual(self.request('/v1/review-batches', submission)[0], 400)

    def test_errors_do_not_return_source_content(self):
        def fail(*_):
            raise RuntimeError('PRIVATE SOURCE CONTENT')
        self.fixture.collector.fetch = fail
        status, body = self.request('/v1/routing-candidates')
        self.assertEqual(status, 500)
        self.assertNotIn('PRIVATE', json.dumps(body))


if __name__ == '__main__':
    unittest.main()
