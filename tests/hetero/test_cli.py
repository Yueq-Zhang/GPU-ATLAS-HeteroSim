from frontend.hetero.cli import main


def test_validate_command(capsys) -> None:
    result = main(
        [
            "validate",
            "--config",
            "configs/hetero/experiments/m0_smoke.yaml",
        ]
    )
    assert result == 0
    assert "simulation_input_key=" in capsys.readouterr().out

