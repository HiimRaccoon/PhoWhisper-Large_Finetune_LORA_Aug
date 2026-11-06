import os
import pytest


@pytest.mark.parametrize('relative', ['../escape.wav', '/outside.wav', 'C:\\outside.wav', 'train_environment/x', 'CS406/x'])
def test_paths_reject_escapes_and_forbidden_state(tmp_path, relative):
    from asr.paths import resolve_beneath
    with pytest.raises(ValueError):
        resolve_beneath(tmp_path, relative)


@pytest.mark.linux
def test_symlink_cannot_escape_root(tmp_path):
    from asr.paths import resolve_beneath
    root=tmp_path/'root'; root.mkdir()
    target=tmp_path/'outside'; target.mkdir()
    os.symlink(target,root/'link',target_is_directory=True)
    with pytest.raises(ValueError):
        resolve_beneath(root,'link/x.wav')
