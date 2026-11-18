"""Filesystem-only operational publisher tests; never construct a READY release."""
import importlib.util
import os
from pathlib import Path
import stat
import subprocess
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def publisher():
    import publish_hsrl_v6_parallel
    return publish_hsrl_v6_parallel


def test_standalone_publisher_exists():
    assert importlib.util.find_spec('publish_hsrl_v6_parallel'), 'standalone publisher missing'


def fixture_tree(root):
    root.mkdir()
    for number in range(24):
        path = root / 'private' / str(number) / 'quality' / 'bytes'
        path.parent.mkdir(parents=True)
        path.write_bytes(bytes([number]) * 300)
        path.chmod(0o640)
        os.utime(path, ns=(1234567890000000000, 1234567890000000000))
    (root / 'empty').mkdir()
    return root


def test_parallel_copy_preserves_all_bytes_metadata_empty_dirs_and_closure(tmp_path):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    os.setxattr(source / 'private/0/quality/bytes', 'user.publisher_test', b'provenance')
    target = tmp_path / 'copy'
    files = p.copy_parallel(source, target, 8)
    assert len(files) == 24
    assert (target / 'empty').is_dir()
    assert os.getxattr(target / 'private/0/quality/bytes', 'user.publisher_test') == b'provenance'
    for name in files:
        left, right = source / name, target / name
        assert left.read_bytes() == right.read_bytes()
        assert left.stat().st_ino != right.stat().st_ino
        assert stat.S_IMODE(left.stat().st_mode) == stat.S_IMODE(right.stat().st_mode)
        assert left.stat().st_mtime_ns == right.stat().st_mtime_ns
    p.check_closure(target, files, 8)
    (target / 'unexpected.py').write_text('raise RuntimeError()')
    with pytest.raises(ValueError, match='inventory'):
        p.check_closure(target, files, 8)


@pytest.mark.parametrize('kind', ['symlink', 'fifo', 'directory_link'])
def test_copy_rejects_links_and_special_files_without_following(tmp_path, kind):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'secret').write_text('private')
    bad = source / 'private' / '0' / 'quality' / 'bad'
    if kind == 'fifo':
        os.mkfifo(bad)
    else:
        bad.symlink_to(outside if kind == 'directory_link' else outside / 'secret')
    with pytest.raises((ValueError, OSError)):
        p.copy_parallel(source, tmp_path / 'copy', 8)
    assert (outside / 'secret').read_text() == 'private'


def test_worker_error_waits_for_other_workers_to_close(tmp_path):
    p = publisher()
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def work(number):
        if number == 0:
            started.set()
            release.wait(2)
            (tmp_path / 'closed').write_text('closed')
            finished.set()
        else:
            assert started.wait(2)
            release.set()
            raise ValueError('worker_failure')
    with pytest.raises(ValueError, match='worker_failure'):
        p.parallel_each(work, range(2), 2)
    assert finished.is_set()
    assert (tmp_path / 'closed').read_text() == 'closed'


def test_bounded_parallel_work_accounts_for_every_input():
    p = publisher()
    barrier, lock = threading.Barrier(8), threading.Lock()
    active = peak = 0
    def work(number):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        if number < 8:
            barrier.wait(timeout=2)
        with lock:
            active -= 1
        return number
    assert sorted(p.parallel_each(work, range(100), 8)) == list(range(100))
    assert peak == 8


def test_copy_detects_source_change_during_real_read(tmp_path, monkeypatch):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    original = p.shutil.copyfileobj
    def copy_and_change(reader, writer, *args):
        original(reader, writer, *args)
        os.utime(reader.fileno(), ns=(1, 1))
    monkeypatch.setattr(p.shutil, 'copyfileobj', copy_and_change)
    with pytest.raises(ValueError, match='source_changed'):
        p.copy_parallel(source, tmp_path / 'copy', 8)


def test_copy_failure_retains_private_stage_and_unchanged_source(tmp_path):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    os.mkfifo(source / 'private/0/quality/failure')
    private = tmp_path / '.private-publication'
    private.mkdir(mode=0o700)
    with pytest.raises(ValueError, match='special'):
        p.copy_parallel(source, private / 'artifact', 8)
    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert (private / 'artifact').is_dir()
    assert stat.S_ISFIFO((source / 'private/0/quality/failure').stat().st_mode)
    assert not (source / 'READY.json').exists()
    assert not (private / 'artifact/READY.json').exists()


def test_parallel_copy_rejects_root_symlink_and_existing_target(tmp_path):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    link = tmp_path / 'link'
    link.symlink_to(source)
    with pytest.raises(OSError):
        p.copy_parallel(link, tmp_path / 'copy', 8)
    target = tmp_path / 'target'
    target.mkdir()
    with pytest.raises(FileExistsError):
        p.copy_parallel(source, target, 8)
    assert not list(target.iterdir())


def test_closure_rejects_missing_file_and_deep_symlink(tmp_path):
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    names = {path.relative_to(source).as_posix() for path in source.rglob('*') if path.is_file()}
    victim = source / next(iter(names))
    victim.unlink()
    with pytest.raises(ValueError, match='inventory'):
        p.check_closure(source, names, 8)
    victim.symlink_to(tmp_path / 'missing')
    with pytest.raises((OSError, ValueError)):
        p.check_closure(source, names, 8)


def test_original_dataset_hash_checks_visit_every_file_and_reject_corruption(tmp_path):
    from hsrl_v5.dataset import Dataset
    from hsrl_v5.io import file_hash
    p = publisher()
    source = fixture_tree(tmp_path / 'source')
    view = Dataset.__new__(Dataset)
    view.root = source
    view.files = {path.relative_to(source).as_posix(): file_hash(path)
                  for path in source.rglob('*') if path.is_file()}
    assert p.verify_files(view, 8) == len(view.files) == 24
    for name in view.files:
        path = source / name
        original = path.read_bytes()
        path.write_bytes(b'corrupt')
        with pytest.raises(ValueError, match='identity_changed'):
            p.verify_files(view, 8)
        path.write_bytes(original)


@pytest.mark.parametrize('kind', ['empty', 'symlink'])
def test_existing_output_refused_before_candidate_read(tmp_path, kind):
    p = publisher()
    out = tmp_path / 'out'
    if kind == 'empty':
        out.mkdir()
    else:
        out.symlink_to(tmp_path / 'absent')
    with pytest.raises(FileExistsError):
        p.publish(tmp_path / 'missing_candidate', out, tmp_path / 'context', tmp_path / 'runtime', 8)
    assert out.is_symlink() if kind == 'symlink' else out.is_dir()


def test_original_atomic_install_refuses_existing_empty_directory(tmp_path):
    from hsrl_v5.release import install_no_replace
    source, target = tmp_path / 'source', tmp_path / 'target'
    source.mkdir()
    target.mkdir()
    (source / 'evidence').write_text('retained')
    with pytest.raises(FileExistsError):
        install_no_replace(source, target)
    assert (source / 'evidence').read_text() == 'retained'
    assert not list(target.iterdir())


def test_worker_bound_rejected():
    p = publisher()
    for count in (0, 9):
        with pytest.raises(ValueError, match='workers'):
            p.parallel_each(lambda x: x, [], count)


@pytest.mark.parametrize('kind', ['symlink', 'fifo'])
def test_manifest_rejects_unsafe_type_before_read(tmp_path, kind):
    p = publisher()
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    path = candidate / 'manifest.json'
    if kind == 'fifo':
        os.mkfifo(path)
    else:
        path.symlink_to(tmp_path / 'absent')
    with pytest.raises((ValueError, OSError)):
        p.publish(candidate, tmp_path / 'out', tmp_path / 'context', tmp_path / 'runtime')
    assert not (tmp_path / 'out').exists()


def test_real_cli_refuses_existing_output_and_has_no_side_effects(tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    evidence = out / 'evidence'
    evidence.write_text('preserved')
    script = Path(__file__).resolve().parents[1] / 'scripts/publish_hsrl_v6_parallel.py'
    result = subprocess.run([sys.executable, '-B', str(script), '--candidate', str(tmp_path / 'missing'),
                             '--out', str(out), '--context-audit', str(tmp_path / 'context'),
                             '--runtime-audit', str(tmp_path / 'runtime')],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0
    assert 'publish_destination_exists' in result.stderr
    assert evidence.read_text() == 'preserved'
    assert sorted(path.name for path in tmp_path.iterdir()) == ['out']
