import importlib.util

from stable_asr.doctor import run_doctor


def test_doctor_accepts_repository_configs() -> None:
    report = run_doctor()

    assert report.ok
    assert isinstance(report.release_environment_ready, bool)
    assert any(check.name == "benchmark_suite" and check.ok for check in report.checks)
    assert any(check.name == "roadmap" and check.ok for check in report.checks)
    assert any(check.name == "asr_collections" and check.ok for check in report.checks)
    assert any(check.name == "final_run" and check.ok for check in report.checks)


def test_doctor_final_file_readiness_is_optional_and_reports_missing_inputs() -> None:
    report = run_doctor(check_final_files=True)

    assert report.ok
    assert not report.final_inputs_ready
    assert "final_inputs_ready: NO" in report.to_text()
    assert "required input(s) missing" in report.to_text()


def test_doctor_release_environment_check_matches_optional_dependencies() -> None:
    report = run_doctor(check_release_env=True)
    expected = _has_import("torch") and _has_working_lance()

    assert report.ok is expected
    assert report.release_environment_ready is expected
    assert "release/environment" in report.to_text()


def _has_import(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _has_working_lance() -> bool:
    if importlib.util.find_spec("lance") is None:
        return False
    try:
        import lance
    except Exception:
        return False
    return hasattr(lance, "dataset") and hasattr(lance, "write_dataset")
