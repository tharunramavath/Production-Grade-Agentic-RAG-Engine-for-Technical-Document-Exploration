from typer.testing import CliRunner
from src.interfaces.cli.main import cli

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Agentic Research Assistant CLI" in result.stdout
    assert "ingest" in result.stdout
    assert "search" in result.stdout
    assert "ask" in result.stdout
    assert "serve" in result.stdout
    assert "ui" in result.stdout


def test_cli_search_empty():
    result = runner.invoke(cli, ["search", "nonexistent_query_term_12345", "--top-k", "2"])
    assert result.exit_code == 0
