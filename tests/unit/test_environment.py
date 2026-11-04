import pytest

from repath_evaluation.cli import build_parser


def test_environment_is_only_created_on_explicit_command(monkeypatch, tmp_path):
    from repath_evaluation import environment

    output = tmp_path / "runtime_environment.txt"
    monkeypatch.setattr(environment, "capture_environment", lambda: "=== DATE ===\nnow\n")
    args = build_parser().parse_args(["environment", "--output", str(output)])
    assert not output.exists()
    args.func(args)
    assert output.read_text() == "=== DATE ===\nnow\n"
    with pytest.raises(SystemExit, match="already exists"):
        args.func(args)


def test_environment_default_stays_under_results():
    args = build_parser().parse_args(["environment"])
    assert args.output == "results/runtime_environment.txt"
