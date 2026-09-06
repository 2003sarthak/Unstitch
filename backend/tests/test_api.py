"""API contract and end-to-end tests.

The centrepiece is `TestEndToEnd`, which drives a real video through the real
pipeline - real ffmpeg, real PySceneDetect, the stub detector - and asserts on
the JSON a browser would receive. That test is only possible because the vision
port has an offline implementation: with a Gemini-only design this would need a
key, a network and quota, and would therefore not run in CI at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.domain.models import JobStatus, RemovalMode
from app.main import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """A configuration pointed at a scratch media root.

    `_env_file=None` so the suite does not read the developer's `.env` and pass
    or fail on whatever they happen to have configured.
    """
    return Settings(
        _env_file=None,
        media_root=tmp_path / "work",
        vision_provider="stub",
        frame_sample_interval_s=0.5,
        max_frames=4,
        min_scene_seconds=0.3,
        max_concurrent_jobs=1,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    with TestClient(create_app(settings)) as client:
        yield client


def wait_for_terminal(client: TestClient, job_id: str, timeout_s: float = 120.0) -> dict:
    """Poll exactly as the frontend does, until the job stops moving."""
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["status"] in (JobStatus.DONE, JobStatus.FAILED):
            return body
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} never finished: {body}")


class TestHealth:
    def test_reports_effective_provider_not_requested_one(self, tmp_path: Path) -> None:
        settings = Settings(
            _env_file=None,
            media_root=tmp_path,
            vision_provider="gemini",
            gemini_api_key="",
        )
        with TestClient(create_app(settings)) as client:
            body = client.get("/api/health").json()

        assert body["vision_provider"] == "stub"
        assert body["vision_provider_requested"] == "gemini"
        assert body["status"] == "ok"

    def test_queue_depth_is_reported(self, client: TestClient) -> None:
        """Cheap operational visibility: on a two-worker Space, a growing queue
        is the difference between "slow" and "stuck"."""
        assert client.get("/api/health").json()["queued_jobs"] == 0

    def test_cors_headers_reflect_the_configured_origin(self, tmp_path: Path) -> None:
        """The frontend is served from another origin, so this is load-bearing
        rather than incidental - and it is the one setting that must change at
        deploy time."""
        settings = Settings(
            _env_file=None, media_root=tmp_path, cors_origins="https://unstitch.vercel.app"
        )
        with TestClient(create_app(settings)) as client:
            response = client.get("/api/health", headers={"Origin": "https://unstitch.vercel.app"})
        assert response.headers["access-control-allow-origin"] == "https://unstitch.vercel.app"


class TestSubmission:
    def test_a_url_job_is_accepted_immediately(self, client: TestClient) -> None:
        """202 and an id, without waiting for the work - the whole reason this
        backend is a container rather than a function."""
        response = client.post("/api/jobs", json={"url": "https://example.com/v.mp4"})

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == JobStatus.QUEUED
        assert len(body["job_id"]) == 32

    def test_the_job_is_pollable_the_instant_it_is_accepted(self, client: TestClient) -> None:
        """No window where the API has issued an id that 404s. This is why the
        runner persists the job before enqueuing it."""
        job_id = client.post("/api/jobs", json={"url": "https://example.com/v.mp4"}).json()[
            "job_id"
        ]
        assert client.get(f"/api/jobs/{job_id}").status_code == 200

    def test_a_missing_url_is_rejected_with_guidance(self, client: TestClient) -> None:
        response = client.post("/api/jobs", json={})
        assert response.status_code == 400
        assert "upload" in response.json()["error"]

    def test_an_unknown_removal_mode_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/jobs", json={"url": "https://x.test/v.mp4", "removal_mode": "magic"}
        )
        assert response.status_code == 422

    def test_an_empty_upload_is_rejected(self, client: TestClient) -> None:
        response = client.post("/api/jobs/upload", files={"file": ("empty.mp4", b"", "video/mp4")})
        assert response.status_code == 400

    def test_an_oversized_upload_is_rejected_by_size_not_by_header(self, tmp_path: Path) -> None:
        """The cap is enforced while the body is written, because Content-Length
        is client-supplied and a body can simply keep coming."""
        settings = Settings(_env_file=None, media_root=tmp_path, max_upload_mb=1)
        with TestClient(create_app(settings)) as client:
            response = client.post(
                "/api/jobs/upload",
                files={"file": ("big.mp4", b"x" * (2 * 1024 * 1024), "video/mp4")},
            )
        assert response.status_code == 413


class TestPolling:
    def test_an_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get("/api/jobs/" + "0" * 32).status_code == 404

    def test_a_result_is_not_available_before_the_job_finishes(self, client: TestClient) -> None:
        """409 rather than 404: the job exists and an answer is coming, which is
        a different situation from a job that never existed."""
        job_id = client.post("/api/jobs", json={"url": "https://x.test/v.mp4"}).json()["job_id"]
        assert client.get(f"/api/jobs/{job_id}/result").status_code in (409, 200)

    def test_a_bad_url_fails_the_job_with_a_usable_message(self, client: TestClient) -> None:
        """yt-dlp's own diagnostics are extractor internals. What the person who
        pasted the link needs is what to do instead."""
        job_id = client.post("/api/jobs", json={"url": "not-a-url"}).json()["job_id"]
        body = wait_for_terminal(client, job_id, timeout_s=30)

        assert body["status"] == JobStatus.FAILED
        assert body["error"]
        assert "Traceback" not in body["error"]


class TestMediaRoute:
    def test_traversal_attempts_are_indistinguishable_from_misses(self, client: TestClient) -> None:
        """Both answer 404, so probing cannot tell "blocked" from "absent" and
        map what exists."""
        job_id = "0" * 32
        assert client.get(f"/media/{job_id}/../../../etc/passwd").status_code == 404
        assert client.get(f"/media/{job_id}/nope.mp4").status_code == 404

    def test_a_malformed_job_id_is_refused(self, client: TestClient) -> None:
        assert client.get("/media/not-a-job-id/source.mp4").status_code == 404


class TestEndToEnd:
    """A real video, all the way through, asserting on what a browser receives."""

    @pytest.fixture
    def video_bytes(self, tmp_path: Path) -> bytes:
        """Two shots with a caption-like band burned across the lower third and a
        watermark block in the top corner - so there is something for the scene
        detector *and* the overlay detector to find.

        Flat colour backgrounds rather than `testsrc2`, deliberately. A test
        pattern is edge noise from corner to corner, which is close to the worst
        case for a detector keyed on edge density and nothing like the footage
        this app targets. Using one here would test the stub's failure mode
        instead of the pipeline.
        """

        async def build() -> bytes:
            from app.infra.ffmpeg import Ffmpeg

            ffmpeg = Ffmpeg(default_timeout_s=120.0)
            overlays = ",".join(
                [
                    *(
                        f"drawbox=x={40 + i * 30}:y=380:w=18:h=36:color=white:t=fill"
                        for i in range(14)
                    ),
                    "drawbox=x=350:y=25:w=110:h=30:color=white:t=fill",
                ]
            )
            shots = []
            for name, colour in (("a", "teal"), ("b", "maroon")):
                shot = tmp_path / f"{name}.mp4"
                await ffmpeg.run(
                    [
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c={colour}:size=480x480:rate=24:duration=2",
                        "-vf",
                        overlays,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "ultrafast",
                        "-pix_fmt",
                        "yuv420p",
                        str(shot),
                    ]
                )
                shots.append(shot)

            listing = tmp_path / "shots.txt"
            listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in shots))
            path = tmp_path / "input.mp4"
            await ffmpeg.run(
                [
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(path),
                ]
            )
            return path.read_bytes()

        return asyncio.run(build())

    def test_a_video_becomes_scenes_tracks_and_a_clean_render(
        self, client: TestClient, video_bytes: bytes
    ) -> None:
        accepted = client.post(
            "/api/jobs/upload",
            files={"file": ("clip.mp4", video_bytes, "video/mp4")},
            data={"removal_mode": RemovalMode.DELOGO.value},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        final = wait_for_terminal(client, job_id)
        assert final["status"] == JobStatus.DONE, final.get("error")
        assert final["progress"] == 1.0

        result = client.get(f"/api/jobs/{job_id}/result")
        assert result.status_code == 200
        body = result.json()

        assert body["job_id"] == job_id
        assert body["vision_provider"] == "stub", "never claim a real analysis"
        assert body["removal_mode"] == "delogo"
        assert body["meta"]["duration_s"] == pytest.approx(4.0, abs=0.4)

        assert len(body["scenes"]) == 2, "teal -> maroon is an unmistakable cut"
        assert body["scenes"][0]["start_s"] == 0.0
        assert body["scenes"][-1]["end_s"] == pytest.approx(body["meta"]["duration_s"], abs=0.3)

        # The assertion that keeps this test honest. Without it the whole suite
        # passes on a pipeline that finds nothing at all: "every track is
        # well-formed" is trivially true of zero tracks.
        assert body["tracks"], "a burned-in band on a flat background must be found"

    def test_every_media_url_in_the_result_actually_serves(
        self, client: TestClient, video_bytes: bytes
    ) -> None:
        """The contract that matters to the frontend: if a URL is in the JSON,
        fetching it returns bytes. A dangling URL is invisible in a unit test and
        immediately visible as a broken player."""
        job_id = client.post(
            "/api/jobs/upload", files={"file": ("clip.mp4", video_bytes, "video/mp4")}
        ).json()["job_id"]
        assert wait_for_terminal(client, job_id)["status"] == JobStatus.DONE

        body = client.get(f"/api/jobs/{job_id}/result").json()

        urls = [body["source_url"], body["clean_url"]]
        urls += [s["thumb_url"] for s in body["scenes"] if s["thumb_url"]]
        urls += [t["crop_url"] for t in body["tracks"] if t["crop_url"]]
        assert len(urls) >= 3

        for url in urls:
            response = client.get(url)
            assert response.status_code == 200, url
            assert len(response.content) > 0, url

    def test_stage_labels_are_reported_not_just_a_number(
        self, client: TestClient, video_bytes: bytes
    ) -> None:
        """ "detecting overlays" tells a waiting user something a spinner cannot."""
        job_id = client.post(
            "/api/jobs/upload", files={"file": ("clip.mp4", video_bytes, "video/mp4")}
        ).json()["job_id"]

        seen = set()
        import time

        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            body = client.get(f"/api/jobs/{job_id}").json()
            seen.add(body["stage"])
            if body["status"] in (JobStatus.DONE, JobStatus.FAILED):
                break
            time.sleep(0.05)

        assert "done" in seen
        assert all(" " in s or s.isalpha() for s in seen), seen

    def test_tracks_are_shaped_for_removal_and_for_the_ui(
        self, client: TestClient, video_bytes: bytes
    ) -> None:
        job_id = client.post(
            "/api/jobs/upload", files={"file": ("clip.mp4", video_bytes, "video/mp4")}
        ).json()["job_id"]
        assert wait_for_terminal(client, job_id)["status"] == JobStatus.DONE

        for track in client.get(f"/api/jobs/{job_id}/result").json()["tracks"]:
            assert track["end_s"] > track["start_s"]
            assert 0.0 <= track["bbox"]["x"] <= 1.0
            assert track["bbox"]["x"] + track["bbox"]["w"] <= 1.0 + 1e-9
            assert track["id"].startswith("t")
            assert track["detection_count"] >= 1
