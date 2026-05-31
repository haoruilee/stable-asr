from stable_asr.doctor import run_doctor


def test_doctor_accepts_repository_configs() -> None:
    report = run_doctor()

    assert report.ok
    assert any(check.name == "benchmark_suite" and check.ok for check in report.checks)
    assert any(check.name == "roadmap" and check.ok for check in report.checks)
    assert any(check.name == "final_run" and check.ok for check in report.checks)


def test_doctor_final_file_readiness_is_optional_and_reports_missing_inputs() -> None:
    report = run_doctor(check_final_files=True)

    assert report.ok
    assert not report.final_inputs_ready
    assert "final_inputs_ready: NO" in report.to_text()
    assert "required input(s) missing" in report.to_text()
