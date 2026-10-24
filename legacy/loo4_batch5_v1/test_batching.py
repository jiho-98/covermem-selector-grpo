"""CPU contract tests for schedule, bank isolation and atomic batch recovery."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from batching import build_batches, history_order, initial_state, validate_state, advance_state
import checkpoint


def sessions_for(lengths):
    return [{'history_id': f'h{h}', 'session_id': f's{i}',
             'start': i * 4, 'mid': i * 4 + 2, 'end': i * 4 + 4}
            for h, length in enumerate(lengths) for i in range(length)]


def end_banks(batch, number):
    return {s['history_id']: [[{'history': s['history_id'], 'stream': k,
                               'batch': number}] for k in range(4)] for s in batch}


class BatchContracts(unittest.TestCase):
    def test_actual_session_counts_and_round_robin_tail(self):
        sessions = sessions_for([29, 28, 30, 25, 30])
        batches = build_batches(sessions)
        self.assertEqual(len(batches), 30)
        self.assertEqual(sum(map(len, batches)), 142)
        self.assertEqual([len(b) for b in batches], [5] * 25 + [4] * 3 + [3, 2])
        for i, batch in enumerate(batches):
            self.assertEqual(len({s['history_id'] for s in batch}), len(batch))
            self.assertTrue(all(s['session_id'] == f's{i}' for s in batch))

    def test_chronological_input_rejected(self):
        sessions = sessions_for([2, 1])
        sessions[1]['start'] = 9
        with self.assertRaises(ValueError):
            build_batches(sessions)

    def test_partial_smoke_limit_is_batch_boundary(self):
        batches = build_batches(sessions_for([3] * 5), max_sessions=7)
        self.assertEqual([len(b) for b in batches], [5, 2])
        self.assertEqual([s['history_id'] for s in batches[1]], ['h0', 'h1'])

    def test_separate_banks_zero_signal_and_resume(self):
        sessions = sessions_for([2, 1, 2])
        histories = history_order(sessions)
        batches = build_batches(sessions)
        state = initial_state(histories, batches, 4)
        first = advance_state(state, batches[0], end_banks(batches[0], 0), False)
        validate_state(first, histories, batches, 4)
        self.assertEqual(first['updates'], 0)
        self.assertEqual(first['completed_sessions'], 3)
        self.assertEqual(state['completed_sessions'], 0)
        self.assertEqual(state['banks_by_history']['h0'], [[], [], [], []])
        second = advance_state(first, batches[1], end_banks(batches[1], 1), True)
        validate_state(second, histories, batches, 4)
        self.assertEqual(second['updates'], 1)
        self.assertEqual(second['completed_sessions'], 5)
        self.assertEqual(second['banks_by_history']['h1'], first['banks_by_history']['h1'])
        self.assertEqual(second['banks_by_history']['h0'][3][0]['stream'], 3)
        bad = deepcopy(second)
        bad['session_cursors']['h0'] -= 1
        with self.assertRaises(ValueError):
            validate_state(bad, histories, batches, 4)
        with self.assertRaises(ValueError):
            validate_state(second, histories, batches, 2)

    def test_repeated_history_in_batch_rejected(self):
        sessions = sessions_for([2])
        batches = build_batches(sessions)
        state = initial_state(['h0'], batches, 1)
        with self.assertRaises(ValueError):
            advance_state(state, sessions, end_banks(sessions, 0), True)

    def test_atomic_checkpoint_failure_and_retry(self):
        sessions = sessions_for([2, 2])
        histories = history_order(sessions)
        batches = build_batches(sessions)
        state = initial_state(histories, batches, 1)
        first = advance_state(state, batches[0], end_banks(batches[0], 0), True)
        second = advance_state(first, batches[1], end_banks(batches[1], 1), True)
        def payload(directory):
            (directory / 'adapter').mkdir()
            (directory / 'adapter/weights.bin').write_bytes(b'fixture')
            (directory / 'training.pt').write_bytes(b'optimizer-and-rng-fixture')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            p1 = checkpoint.publish(root, 1, 'fixture-fingerprint', first, payload)
            self.assertEqual(checkpoint.latest(root, 'fixture-fingerprint')[1], first)
            original_atomic = checkpoint.atomic
            def crash_before_pointer(path, value):
                if Path(path).name == 'latest.json':
                    raise OSError('simulated publication interruption')
                return original_atomic(path, value)
            with patch.object(checkpoint, 'atomic', crash_before_pointer):
                with self.assertRaises(OSError):
                    checkpoint.publish(root, 2, 'fixture-fingerprint', second, payload)
            self.assertTrue(p1.exists())
            self.assertEqual(checkpoint.latest(root, 'fixture-fingerprint')[1], first)
            p2 = checkpoint.publish(root, 2, 'fixture-fingerprint', second, payload)
            self.assertEqual(checkpoint.latest(root, 'fixture-fingerprint')[1], second)
            self.assertFalse(p1.exists())
            (p2 / 'training.pt').write_bytes(b'corruption')
            with self.assertRaises(RuntimeError):
                checkpoint.latest(root, 'fixture-fingerprint')

    def test_incomplete_batch_state_not_publishable(self):
        sessions = sessions_for([1])
        state = initial_state(['h0'], build_batches(sessions), 1)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                checkpoint.publish(Path(temporary), 1, 'fixture', state, lambda _: None)


if __name__ == '__main__':
    unittest.main()
