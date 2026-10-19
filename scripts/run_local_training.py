"""Run one finite local training command with logs, timeout and macOS sleep prevention."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def run(command, output, *, cwd=ROOT, timeout_seconds=48 * 3600, keep_awake=True):
    if not command or any(not isinstance(arg, str) or not arg for arg in command):
        raise ValueError('Provide a nonempty command argument list')
    if timeout_seconds <= 0:
        raise ValueError('Timeout must be positive')
    output = Path(output).resolve()
    output.mkdir(parents=True, mode=0o700)  # Existing runs must not be overwritten.
    actual = list(command)
    if keep_awake and sys.platform == 'darwin' and shutil.which('caffeinate'):
        actual = [shutil.which('caffeinate'), '-i', '-s'] + actual
    record = {'command': command, 'actual_command': actual, 'cwd': str(Path(cwd).resolve()),
              'runner_pid': os.getpid(), 'started_at': timestamp(), 'status': 'starting',
              'timeout_seconds': timeout_seconds, 'thread_limit': 4}

    def save():
        with tempfile.NamedTemporaryFile('w', dir=output, delete=False) as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            name = stream.name
        os.replace(name, output / 'run.json')

    save()
    environment = dict(os.environ)
    environment['PYTHONUNBUFFERED'] = '1'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[name] = '4'
    process = None
    started = time.monotonic()
    code = 1
    try:
        fd = os.open(output / 'training.log', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, 'w') as log:
            process = subprocess.Popen(actual, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record.update(child_pid=process.pid, status='running')
            save()
            try:
                code = process.wait(timeout=timeout_seconds)
                record.update(status='completed' if code == 0 else 'failed', returncode=code)
            except subprocess.TimeoutExpired:
                record['status'] = 'timed_out'
                code = 124
    except (OSError, KeyboardInterrupt) as error:
        record['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        record['error_type'] = type(error).__name__
        code = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        record.update(returncode=code, finished_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        save()
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New ignored output directory for this execution')
    parser.add_argument('--timeout-hours', type=float, default=48)
    parser.add_argument('--allow-sleep', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not 0 < args.timeout_hours <= 72:
        parser.error('Provide a command after -- and a timeout between zero and 72 hours')
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        code = run(command, args.output, timeout_seconds=args.timeout_hours * 3600, keep_awake=not args.allow_sleep)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({'output': str(args.output.resolve()), 'returncode': code}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
