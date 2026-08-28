import json
import sys
from pathlib import Path

from frontend.hetero.backends import AtlasArtifact, AtlasBackend, AtlasBackendConfig


def _fake_backend(tmp_path: Path) -> tuple[AtlasBackend, Path, AtlasArtifact]:
    runner = tmp_path / "fake_atlas_runner.py"
    runner.write_text(
        "import argparse,json\n"
        "p=argparse.ArgumentParser()\n"
        "p.add_argument('--chip'); p.add_argument('--operators')\n"
        "p.add_argument('--placement'); p.add_argument('--output')\n"
        "a=p.parse_args()\n"
        "s={'schema_version':'hetero-atlas-native-stats/v1',"
        "'chip_frequency_mhz':1000,'e2e_stats':{"
        "'e2e_cycles':48446,'e2e_energy':0.00581352,'dram_cycles':48446}}\n"
        "open(a.output,'w').write(json.dumps(s))\n",
        encoding="utf-8",
    )
    chip = tmp_path / "chip.yaml"
    operators = tmp_path / "operators.yaml"
    placement = tmp_path / "placement.yaml"
    for path in (chip, operators, placement):
        path.write_text("test: true\n", encoding="utf-8")
    config_path = tmp_path / "atlas_backend.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "hetero-atlas-backend/v1",
                "backend_id": "atlas.atlasim.fake",
                "python_executable": sys.executable,
                "atlas_root": str(tmp_path),
                "adapter_script": runner.name,
                "core_frequency_hz": 1_000_000_000,
                "timeout_seconds": 10,
                "dependency_commits": {"atlas": "b278739"},
            }
        ),
        encoding="utf-8",
    )
    backend = AtlasBackend(AtlasBackendConfig.load(config_path))
    artifact = AtlasArtifact(operators, placement)
    return backend, chip, artifact


def test_atlas_total_duration_and_equivalence_record(tmp_path: Path) -> None:
    backend, chip, artifact = _fake_backend(tmp_path)
    result = backend.run(chip, artifact, tmp_path / "run")

    assert result.cycles == 48446
    assert result.duration_fs == 48_446_000_000
    assert result.energy_j == 0.00581352
    assert backend.descriptor().supported_duration_semantics == ("total",)
    assert backend.descriptor().supported_exports == ()

    record_path = backend.qualify(chip, artifact, tmp_path / "qualification")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "passed"
    assert record["comparison"]["cycles"] == [48446, 48446]
    assert record["timing_ownership"]["external_ramulator2"] is False


def test_atlas_relative_artifacts_survive_backend_cwd_change(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    backend, chip, artifact = _fake_backend(project)
    monkeypatch.chdir(project)

    relative_artifact = AtlasArtifact(
        artifact.operator_list.relative_to(project),
        artifact.placement_map.relative_to(project),
    )
    result = backend.run(
        chip.relative_to(project), relative_artifact, project / "relative_run"
    )

    assert result.cycles == 48_446
    command = json.loads((project / "relative_run" / "command.json").read_text())
    assert Path(command["argv"][3]).is_absolute()
    assert Path(command["argv"][5]).is_absolute()
    assert Path(command["argv"][7]).is_absolute()
