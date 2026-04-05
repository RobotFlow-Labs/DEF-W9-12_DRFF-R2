from __future__ import annotations

from typer.testing import CliRunner

from anima_drff_r2.cli import app


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ANIMA DRFF-R2" in result.stdout
