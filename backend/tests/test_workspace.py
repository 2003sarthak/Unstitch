"""Tests for the job workspace.

Two things here are worth more than the rest: the naming scheme (because three
separate modules depend on agreeing about it) and the path-escape checks
(because `/media/{job}/{rest}` is the one route that turns client-supplied text
into a filesystem path).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from app.infra.workspace import (
    InvalidJobIdError,
    PathEscapeError,
    Workspace,
    WorkspaceManager,
    validate_job_id,
)

JOB = "0123456789abcdef0123456789abcdef"


@pytest.fixture
def manager(tmp_path: Path) -> WorkspaceManager:
    return WorkspaceManager(tmp_path / "work")


class TestLayout:
    def test_indexed_names_are_zero_padded_so_they_sort_chronologically(
        self, manager: WorkspaceManager
    ) -> None:
        """Unpadded names would sort scene 10 before scene 2, and the timeline
        strip renders in whatever order it receives."""
        ws = manager.open(JOB)
        names = sorted(ws.scene_clip(i).name for i in (1, 2, 10))
        assert names == ["s001.mp4", "s002.mp4", "s010.mp4"]

    def test_clean_scene_clips_sit_beside_their_originals(self, manager: WorkspaceManager) -> None:
        ws = manager.open(JOB)
        assert ws.scene_clip(3).name == "s003.mp4"
        assert ws.scene_clip(3, clean=True).name == "s003_clean.mp4"
        assert ws.scene_clip(3).parent == ws.scene_clip(3, clean=True).parent

    def test_every_artefact_lives_under_the_job_root(self, manager: WorkspaceManager) -> None:
        ws = manager.open(JOB)
        paths = [
            ws.original,
            ws.source,
            ws.clean,
            ws.frame(0),
            ws.scene_clip(0),
            ws.scene_thumb(0),
            ws.track_crop(0),
            ws.rebuild(1),
        ]
        assert all(p.is_relative_to(ws.root) for p in paths)

    def test_ensure_creates_every_directory_at_once(self, manager: WorkspaceManager) -> None:
        """One mkdir burst at job start, so a full or read-only volume fails
        immediately rather than forty seconds into a render."""
        ws = manager.create(JOB)
        for directory in (
            ws.root,
            ws.frames_dir,
            ws.scenes_dir,
            ws.thumbs_dir,
            ws.crops_dir,
            ws.rebuild_dir,
        ):
            assert directory.is_dir()

    def test_ensure_is_idempotent(self, manager: WorkspaceManager) -> None:
        manager.create(JOB)
        manager.create(JOB)  # a retried job must not blow up on existing dirs


class TestJobIdValidation:
    @pytest.mark.parametrize(
        "bad",
        ["", "short", "../etc", "a/b/c", "job id", "a" * 65, "..", "job;rm -rf"],
    )
    def test_dangerous_or_malformed_ids_are_refused(self, bad: str) -> None:
        with pytest.raises(InvalidJobIdError):
            validate_job_id(bad)

    def test_a_uuid4_hex_is_accepted(self) -> None:
        assert validate_job_id(JOB) == JOB

    def test_manager_refuses_to_open_a_bad_id(self, manager: WorkspaceManager) -> None:
        with pytest.raises(InvalidJobIdError):
            manager.open("../../../etc")


class TestUrlMapping:
    def test_url_is_the_path_relative_to_the_job_root(self, manager: WorkspaceManager) -> None:
        ws = manager.create(JOB)
        assert ws.url_for(ws.clean) == f"/media/{JOB}/clean.mp4"
        assert ws.url_for(ws.scene_thumb(2)) == f"/media/{JOB}/thumbs/s002.jpg"

    def test_urls_use_forward_slashes_on_every_platform(self, manager: WorkspaceManager) -> None:
        """`Path` renders backslashes on Windows; a URL must not."""
        url = manager.create(JOB).url_for(manager.open(JOB).frame(7))
        assert "\\" not in url
        assert url == f"/media/{JOB}/frames/f0007.jpg"

    def test_a_path_outside_the_workspace_has_no_url(self, manager: WorkspaceManager) -> None:
        ws = manager.create(JOB)
        with pytest.raises(PathEscapeError):
            ws.url_for(ws.root.parent / "someone-elses.mp4")


class TestMediaPathResolution:
    """The traversal surface. Two independent defences: the id must match the id
    pattern, and the *resolved* path must still be inside the workspace."""

    def test_a_normal_request_resolves(self, manager: WorkspaceManager) -> None:
        ws = manager.create(JOB)
        assert (
            manager.resolve_media_path(JOB, "thumbs/s001.jpg")
            == (ws.root / "thumbs" / "s001.jpg").resolve()
        )

    @pytest.mark.parametrize(
        "attack",
        [
            "../../../../etc/passwd",
            "..\\..\\..\\windows\\win.ini",
            "thumbs/../../../secrets.env",
            "frames/../../../../.env",
        ],
    )
    def test_traversal_attempts_are_refused(self, manager: WorkspaceManager, attack: str) -> None:
        manager.create(JOB)
        with pytest.raises(PathEscapeError):
            manager.resolve_media_path(JOB, attack)

    def test_traversal_that_stays_inside_is_allowed(self, manager: WorkspaceManager) -> None:
        """`..` is not itself the problem - leaving the workspace is. Rejecting
        the substring rather than checking the resolved path would be a rule that
        looks strict and still misses symlinks."""
        ws = manager.create(JOB)
        resolved = manager.resolve_media_path(JOB, "frames/../thumbs/s000.jpg")
        assert resolved == (ws.thumbs_dir / "s000.jpg").resolve()


class TestSweep:
    """`now` is injected so the suite does not have to wait two hours."""

    def _age(self, path: Path, *, minutes: float, now: float) -> None:
        import os

        stamp = now - minutes * 60
        os.utime(path, (stamp, stamp))

    def test_expired_workspaces_are_deleted(self, manager: WorkspaceManager) -> None:
        now = 1_000_000.0
        stale = manager.create(JOB)
        self._age(stale.root, minutes=180, now=now)

        assert manager.sweep(ttl_minutes=120, now=now) == [JOB]
        assert not stale.root.exists()

    def test_fresh_workspaces_survive(self, manager: WorkspaceManager) -> None:
        now = 1_000_000.0
        fresh = manager.create(JOB)
        self._age(fresh.root, minutes=5, now=now)

        assert manager.sweep(ttl_minutes=120, now=now) == []
        assert fresh.root.is_dir()

    def test_sweep_on_a_missing_root_is_not_an_error(self, tmp_path: Path) -> None:
        """The sweeper runs on a timer and may fire before the first job."""
        assert WorkspaceManager(tmp_path / "never-created").sweep(ttl_minutes=1) == []

    def test_only_directories_are_considered(self, manager: WorkspaceManager) -> None:
        """A loose file in the media root - a stray log, a .gitkeep - is not a
        workspace and must survive a sweep that expires everything else."""
        now = 1_000_000.0
        ws = manager.create(JOB)
        stray = manager.root / "stray.txt"
        stray.write_text("not a workspace")
        for path in (ws.root, stray):
            self._age(path, minutes=180, now=now)

        assert manager.sweep(ttl_minutes=120, now=now) == [JOB]
        assert stray.exists()


def test_workspace_is_a_value_object(manager: WorkspaceManager) -> None:
    """Frozen, so a job's paths cannot be mutated halfway through a pipeline."""
    ws = manager.open(JOB)
    assert isinstance(ws, Workspace)
    with pytest.raises(FrozenInstanceError):
        ws.job_id = "other"  # type: ignore[misc]
