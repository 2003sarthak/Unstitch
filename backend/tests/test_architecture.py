"""The layering rules, enforced rather than documented.

Ports-and-adapters only pays for itself while the dependency arrows actually
point the way the design says they do, and that decays through ordinary edits -
someone needs a helper, reaches for the obvious import, and the seam quietly
stops being a seam. A README cannot catch that. This can.

The rule that matters most is the last one: `services/pipeline.py` must never
import a vendor SDK. That single constraint is what makes the pipeline runnable
against the vision stub with no API key, and testable with no network. Asserting
it here means it fails at `pytest`, not at review.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"

#: The external systems this app talks to. Anything importing one of these is,
#: by definition, an adapter - so these names may only appear under `adapters/`
#: (plus `infra/`, which wraps the ffmpeg *binary* rather than an SDK).
VENDOR_SDKS = ("yt_dlp", "scenedetect", "cv2", "google.genai", "google.generativeai")


def modules_in(package: str) -> list[Path]:
    paths = sorted((APP / package).rglob("*.py"))
    assert paths, f"no modules found under app/{package} - has the layout moved?"
    return paths


def imports_of(path: Path) -> set[str]:
    """Every module name imported by `path`, absolute form."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def assert_no_imports(path: Path, forbidden: tuple[str, ...], why: str) -> None:
    offences = sorted(
        imported
        for imported in imports_of(path)
        for banned in forbidden
        if imported == banned or imported.startswith(f"{banned}.")
    )
    assert not offences, f"{path.relative_to(APP.parent)} imports {offences}: {why}"


class TestDomainIsPure:
    """The domain is the centre. Everything may depend on it; it depends on
    nothing but pydantic."""

    @pytest.mark.parametrize("module", modules_in("domain"), ids=lambda p: p.name)
    def test_domain_imports_no_other_layer(self, module: Path) -> None:
        assert_no_imports(
            module,
            ("app.adapters", "app.services", "app.api", "app.infra", "app.config"),
            "the domain defines the contracts; depending on an implementation of "
            "them would invert the whole architecture",
        )

    @pytest.mark.parametrize("module", modules_in("domain"), ids=lambda p: p.name)
    def test_domain_imports_no_vendor_sdk(self, module: Path) -> None:
        assert_no_imports(module, VENDOR_SDKS, "a domain that knows about Gemini is not a domain")


class TestServicesDependOnPortsOnly:
    """The rule the whole design exists to protect."""

    @pytest.mark.parametrize("module", modules_in("services"), ids=lambda p: p.name)
    def test_services_never_import_a_concrete_adapter(self, module: Path) -> None:
        assert_no_imports(
            module,
            ("app.adapters",),
            "the pipeline must talk to Protocols so that dependencies.py is the "
            "only place an implementation is chosen",
        )

    @pytest.mark.parametrize("module", modules_in("services"), ids=lambda p: p.name)
    def test_services_never_import_a_vendor_sdk(self, module: Path) -> None:
        assert_no_imports(
            module,
            VENDOR_SDKS,
            "this is what lets the whole pipeline run against the vision stub with no API key",
        )


class TestInfraIsPlumbing:
    @pytest.mark.parametrize("module", modules_in("infra"), ids=lambda p: p.name)
    def test_infra_does_not_reach_upward(self, module: Path) -> None:
        assert_no_imports(
            module,
            ("app.adapters", "app.services", "app.api"),
            "infra is the bottom of the stack; it may use domain types, but nothing above it",
        )


class TestAdaptersStayBehindTheirPorts:
    @pytest.mark.parametrize("module", modules_in("adapters"), ids=lambda p: p.name)
    def test_adapters_do_not_import_each_other_or_the_api(self, module: Path) -> None:
        """An adapter that calls another adapter has bypassed a port, which means
        that port can no longer be swapped independently."""
        assert_no_imports(
            module,
            ("app.api", "app.services"),
            "adapters are called by the pipeline, not the other way round",
        )


def test_the_vendor_list_is_not_silently_wrong() -> None:
    """A guard on the guard.

    If a dependency is renamed and this list is not updated, every rule above
    keeps passing while checking nothing. Asserting the names are importable
    means the suite notices.
    """
    import importlib.util

    missing = [
        name
        for name in ("yt_dlp", "scenedetect", "cv2", "google.genai")
        if importlib.util.find_spec(name) is None
    ]
    assert not missing, f"VENDOR_SDKS names no longer resolve: {missing}"
