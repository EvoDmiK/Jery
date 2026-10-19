import json
from pathlib import Path
import sys
import tempfile
import signal
import subprocess
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from run_local_training import run


class LocalTrainingTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name) / 'run'

    def test_command_logs_and_completed_state(self):
        command = [sys.executable, '-c', 'import os; print("training finished", os.environ["OMP_NUM_THREADS"])']
        self.assertEqual(run(command, self.output, keep_awake=False), 0)
        record = json.loads((self.output / 'run.json').read_text())
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['command'], command)
        self.assertIn('training finished 4', (self.output / 'training.log').read_text())
        with self.assertRaises(FileExistsError):
            run(command, self.output)

    def test_failure_and_timeout_do_not_look_completed(self):
        code = run([sys.executable, '-c', 'raise SystemExit(7)'], self.output, keep_awake=False)
        self.assertEqual(code, 7)
        self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'failed')
        timed = self.output.parent / 'timeout'
        code = run([sys.executable, '-c', 'import time; time.sleep(10)'], timed, timeout_seconds=.05, keep_awake=False)
        self.assertEqual(code, 124)
        self.assertEqual(json.loads((timed / 'run.json').read_text())['status'], 'timed_out')

    def test_sigterm_records_interruption_and_stops_child(self):
        script = Path(__file__).resolve().parents[2] / 'scripts/run_local_training.py'
        process = subprocess.Popen([sys.executable, str(script), '--allow-sleep', '--output', str(self.output),
                                    '--', sys.executable, '-c', 'import time; time.sleep(30)'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            deadline = time.monotonic() + 3
            record = {}
            while time.monotonic() < deadline:
                if (self.output / 'run.json').exists():
                    record = json.loads((self.output / 'run.json').read_text())
                    if record['status'] == 'running':
                        break
                time.sleep(.02)
            self.assertEqual(record.get('status'), 'running')
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=3)
            self.assertEqual(process.returncode, 130)
            self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'interrupted')
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()

    def test_missing_executable_records_failure(self):
        self.assertEqual(run(['/nonexistent/jery-model'], self.output, keep_awake=False), 1)
        self.assertEqual(json.loads((self.output / 'run.json').read_text())['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
