import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagnostics.storage import (configure_cache_environment, project_results_root, result_dir,
    require_project_result_dir, register_result_dir, validate_slurm_logs)


class StorageTests(unittest.TestCase):
    def test_direct_entrypoints_redirect_caches_without_changing_home(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ):
            _, storage = self.fixture(folder)
            original_home = os.environ.get('HOME')
            os.environ['TRANSFORMERS_CACHE'] = str(Path.home()/'.cache/old_cache')
            values = configure_cache_environment(storage)
            self.assertEqual(os.environ.get('HOME'), original_home)
            self.assertNotIn('TRANSFORMERS_CACHE', os.environ)
            self.assertTrue(all(Path(v).is_relative_to(storage/'caches') for v in values.values()))
            self.assertTrue(Path(os.environ['HF_HUB_CACHE']).is_dir())

    def fixture(self, folder):
        root = Path(folder)
        repository, storage = root/'repository', root/'project'
        repository.mkdir()
        (storage/'results').mkdir(parents=True)
        return repository, storage

    def test_fresh_repository_results_route_to_project_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            repository, storage = self.fixture(folder)
            path = repository/'results/diagnostics/new_run'
            target = result_dir(path, storage_root=storage, repository_root=repository)
            self.assertEqual(target, storage/'results/diagnostics/new_run')
            self.assertFalse(path.exists())
            self.assertEqual(result_dir(repository/'part2action/results/diagnostics/new_run',
                storage_root=storage, repository_root=repository), target)

    def test_project_paths_and_existing_browsing_alias_are_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            repository, storage = self.fixture(folder)
            target = storage/'results/diagnostics/run'
            target.mkdir(parents=True)
            alias = register_result_dir(target, storage_root=storage, repository_root=repository)
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), target)
            self.assertEqual(result_dir(alias, storage_root=storage, repository_root=repository), target)
            self.assertEqual(register_result_dir(target, storage_root=storage, repository_root=repository), alias)

    def test_arbitrary_home_and_escaping_project_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            repository, storage = self.fixture(folder)
            with self.assertRaises(ValueError):
                result_dir(Path.home()/'unsafe_p2a_output', storage_root=storage, repository_root=repository)
            outside = Path(folder)/'outside'
            outside.mkdir()
            (storage/'results/escape').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                result_dir(repository/'results/escape/new', storage_root=storage, repository_root=repository)
            with self.assertRaises(ValueError):
                require_project_result_dir(outside, storage_root=storage)

    def test_existing_home_results_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            repository, storage = self.fixture(folder)
            path = repository/'results/diagnostics/old_run'
            path.mkdir(parents=True)
            (path/'keep.txt').write_text('existing result')
            with self.assertRaises(ValueError):
                result_dir(path, storage_root=storage, repository_root=repository)
            target = storage/'results/diagnostics/old_run'
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                register_result_dir(target, storage_root=storage, repository_root=repository)
            self.assertEqual((path/'keep.txt').read_text(), 'existing result')

    def test_slurm_requires_both_log_destinations_on_project_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            _, storage = self.fixture(folder)
            valid = f'JobId=1 StdOut={storage}/results/diagnostics/run/job.out StdErr={storage}/logs/job.err'
            self.assertEqual(len(validate_slurm_logs(valid, storage_root=storage)), 2)
            self.assertEqual(len(validate_slurm_logs(valid.replace(f'{storage}/logs/job.err', '/dev/null'), storage_root=storage)), 2)
            for text in [valid.replace(f'{storage}/logs/job.err', str(Path.home()/'bad.err')), 'JobId=1 StdOut=/dev/null']:
                with self.assertRaises(ValueError):
                    validate_slurm_logs(text, storage_root=storage)

    def test_project_results_root_cannot_redirect_outside_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            storage, outside = root/'project', root/'outside'
            storage.mkdir()
            outside.mkdir()
            (storage/'results').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                project_results_root(storage)


if __name__ == '__main__':
    unittest.main()
