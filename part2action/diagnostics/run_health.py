"""Durable process diagnostics using only the standard library."""
import argparse
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import resource
import signal
import sys
import tempfile
import traceback


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name,
                                         suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class TrainingInterrupted(RuntimeError):
    pass


class RunHealth:
    """Catch Python errors/signals; fatal stack dumps cover native crashes/stalls.

    SIGKILL and node loss cannot be caught. The shell exit record and Slurm
    accounting remain independent evidence for those cases.
    """
    def __init__(self, directory, *, stack_dump_seconds=600):
        self.directory = Path(directory)
        self.context = {}
        self.stack_dump_seconds = stack_dump_seconds
        self.handlers = {}

    def event(self, stage, **details):
        self.context.update(stage=stage, **details)
        row = {'time_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
               'job_id': os.environ.get('SLURM_JOB_ID'),
               'host': os.uname().nodename,
               'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               **self.context}
        with (self.directory/'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        atomic_json(self.directory/'heartbeat.json', row)

    def _handle_signal(self, number, frame):
        self.context['signal'] = signal.Signals(number).name
        raise TrainingInterrupted('Training received '+self.context['signal'])

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.fault_log = (self.directory/'fault.log').open('a')
        self.previously_enabled = faulthandler.is_enabled()
        faulthandler.enable(self.fault_log, all_threads=True)
        if self.stack_dump_seconds:
            faulthandler.dump_traceback_later(self.stack_dump_seconds, repeat=True, file=self.fault_log)
        for number in (signal.SIGTERM, signal.SIGUSR1):
            self.handlers[number] = signal.signal(number, self._handle_signal)
        self.event('process_started', argv=sys.argv)
        return self

    def __exit__(self, kind, error, tb):
        try:
            if error is not None and not (isinstance(error, SystemExit) and error.code in (None, 0)):
                trace = ''.join(traceback.format_exception(kind, error, tb))
                with (self.directory/'traceback.log').open('a') as handle:
                    handle.write(trace)
                    handle.flush()
                    os.fsync(handle.fileno())
                self.event('failed', exception_type=kind.__name__, error=str(error))
                atomic_json(self.directory/'failure.json', {
                    'status': 'failed', 'exception_type': kind.__name__, 'error': str(error),
                    'traceback': trace, 'context': self.context})
            else:
                self.event('process_completed')
        except OSError as logging_error:
            print('Could not persist failure diagnostics: '+repr(logging_error), file=sys.stderr, flush=True)
        finally:
            faulthandler.cancel_dump_traceback_later()
            if self.previously_enabled:
                faulthandler.enable(file=sys.__stderr__, all_threads=True)
            else:
                faulthandler.disable()
            self.fault_log.close()
            for number, handler in self.handlers.items():
                signal.signal(number, handler)
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shell-exit', type=int, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.directory/'shell_exit.json', {
        'exit_code': args.shell_exit, 'time_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'host': os.uname().nodename})


if __name__ == '__main__':
    main()
