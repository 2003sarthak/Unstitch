"""Tests for the settings layer.

Every `Settings(...)` here passes `_env_file=None`. Without it pydantic-settings
would still read `backend/.env`, and these tests would pass or fail depending on
what the developer happened to have configured locally - the exact class of flaky
test that makes people stop trusting a suite.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import BACKEND_DIR, Settings
from app.domain.models import RemovalMode, VisionProvider


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestCorsOrigins:
    def test_comma_separated_string_is_split(self) -> None:
        s = make_settings(cors_origins="http://localhost:5173,https://unstitch.vercel.app")
        assert s.cors_origins == ["http://localhost:5173", "https://unstitch.vercel.app"]

    def test_whitespace_and_empty_entries_are_dropped(self) -> None:
        s = make_settings(cors_origins=" http://a.test , , http://b.test ")
        assert s.cors_origins == ["http://a.test", "http://b.test"]

    def test_real_list_passes_through(self) -> None:
        s = make_settings(cors_origins=["http://a.test"])
        assert s.cors_origins == ["http://a.test"]


class TestVisionProviderResolution:
    """The degrade-instead-of-crash policy, pinned so it cannot drift silently."""

    def test_gemini_without_key_degrades_to_stub(self) -> None:
        s = make_settings(vision_provider="gemini", gemini_api_key="")
        assert s.effective_vision_provider is VisionProvider.STUB
        assert s.vision_downgraded is True
        # The requested value is preserved: that is what makes the warning and
        # the health endpoint able to say *what* went wrong.
        assert s.vision_provider is VisionProvider.GEMINI

    def test_gemini_with_key_is_honoured(self) -> None:
        s = make_settings(vision_provider="gemini", gemini_api_key="AIza-not-a-real-key")
        assert s.effective_vision_provider is VisionProvider.GEMINI
        assert s.vision_downgraded is False

    def test_stub_is_never_treated_as_a_downgrade(self) -> None:
        s = make_settings(vision_provider="stub", gemini_api_key="")
        assert s.effective_vision_provider is VisionProvider.STUB
        assert s.vision_downgraded is False


class TestPathHandling:
    def test_blank_cookies_file_becomes_none(self) -> None:
        assert make_settings(ytdlp_cookies_file="").ytdlp_cookies_file is None
        assert make_settings(ytdlp_cookies_file="   ").ytdlp_cookies_file is None

    def test_relative_media_root_anchors_to_backend_dir(self) -> None:
        s = make_settings(media_root="./work")
        assert s.media_root_path == (BACKEND_DIR / "work").resolve()
        assert s.media_root_path.is_absolute()


class TestBounds:
    """Config typos should fail loudly at startup, not silently misbehave later."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("track_iou_threshold", 50),  # percent mistaken for a ratio
            ("min_detection_confidence", -1),
            ("max_frames", 0),
            ("vision_batch_size", 0),
            ("target_height", 120),  # below the detector's useful range
            ("scene_threshold", 0),
            ("mask_padding_pct", 90),  # would mask most of the frame
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: object) -> None:
        with pytest.raises(ValidationError):
            make_settings(**{field: value})


class TestEnums:
    def test_unknown_removal_mode_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(removal_mode="magic-eraser")

    def test_removal_mode_parses_from_string(self) -> None:
        assert make_settings(removal_mode="boxblur").removal_mode is RemovalMode.BOXBLUR


def test_env_example_and_settings_declare_the_same_keys() -> None:
    """`.env.example` is documentation, and documentation rots.

    This asserts the committed example file and the Settings model stay in sync,
    so a new knob cannot be added to one without the other.
    """
    example = (BACKEND_DIR / ".env.example").read_text(encoding="utf-8")
    documented = {
        line.split("=", 1)[0].strip().lower()
        for line in example.splitlines()
        if line and not line.lstrip().startswith("#") and "=" in line
    }
    declared = set(Settings.model_fields)
    assert documented == declared, {
        "in .env.example but not in Settings": sorted(documented - declared),
        "in Settings but undocumented": sorted(declared - documented),
    }
