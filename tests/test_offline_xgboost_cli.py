"""네이티브 모델 학습 없이 CLI 선택과 잘못된 옵션을 거부하는 전달 경로를 검사합니다."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from aml_ml import offline_cli
from aml_ml.offline_rules import R02Policy
from aml_ml.storage import write_json


@pytest.fixture
def cli_case(tmp_path, policy, source, monkeypatch):
    policy_path = tmp_path / "policy.json"
    write_json(policy_path, asdict(policy))
    calls = []
    selected_modules = []

    def load_source(index, selected):
        assert index == tmp_path / "index.json"
        assert selected == ["ibm_hi_small"]
        return [source]

    def training_module(algorithm):
        selected_modules.append(algorithm)

        def train(*args, **kwargs):
            calls.append(("train", args, kwargs))
            return {"status": "VERIFIED", "algorithm": algorithm}

        def verify(*args, **kwargs):
            calls.append(("verify-model", args, kwargs))
            return {"status": "VERIFIED", "algorithm": algorithm}

        return SimpleNamespace(train_evaluate=train, verify_model=verify)

    monkeypatch.setattr(offline_cli, "load_index", load_source)
    monkeypatch.setattr(offline_cli, "_training_module", training_module)
    return SimpleNamespace(
        common=[
            "--assembly-index",
            str(tmp_path / "index.json"),
            "--source",
            "ibm_hi_small",
            "--policy",
            str(policy_path),
            "--scope",
            "FIXTURE_ONLY",
        ],
        dataset=tmp_path / "data",
        output=tmp_path / "model",
        policy_path=policy_path,
        source=source,
        calls=calls,
        selected_modules=selected_modules,
    )


def train_arguments(case):
    return [
        "train",
        *case.common,
        "--dataset",
        str(case.dataset),
        "--output",
        str(case.output),
        "--model-version",
        "fixture-cli",
        "--k",
        "2",
    ]


@pytest.mark.parametrize("selection", [[], ["--algorithm", "numpy-logistic"]])
def test_numpy_default_and_explicit_selection_preserve_existing_arguments(
    cli_case, policy, selection
):
    assert offline_cli.main([*train_arguments(cli_case), *selection]) == 0
    assert cli_case.selected_modules == ["numpy-logistic"]
    name, args, kwargs = cli_case.calls[0]
    assert name == "train"
    assert args == (cli_case.dataset, cli_case.source, policy, cli_case.output)
    assert kwargs == {
        "model_version": "fixture-cli",
        "k": 2,
        "execution_scope": "FIXTURE_ONLY",
        "max_source_rows": 100_000,
    }


def test_xgboost_minimum_options_leave_algorithm_defaults_with_adapter(cli_case):
    assert (
        offline_cli.main([*train_arguments(cli_case), "--algorithm", "xgboost", "--seed", "0"]) == 0
    )
    assert cli_case.selected_modules == ["xgboost"]
    assert cli_case.calls[0][2] == {
        "model_version": "fixture-cli",
        "k": 2,
        "execution_scope": "FIXTURE_ONLY",
        "max_source_rows": 100_000,
        "seed": 0,
    }


def test_xgboost_explicit_hyperparameters_and_source_budget_are_forwarded(cli_case):
    result = offline_cli.main(
        [
            *train_arguments(cli_case),
            "--algorithm",
            "xgboost",
            "--seed",
            "7",
            "--num-boost-round",
            "20",
            "--max-depth",
            "4",
            "--learning-rate",
            "0.2",
            "--min-child-weight",
            "0",
            "--l2",
            "2.5",
            "--early-stopping-rounds",
            "5",
            "--max-source-rows",
            "99",
        ]
    )
    assert result == 0
    assert cli_case.selected_modules == ["xgboost"]
    assert cli_case.calls[0][2] == {
        "model_version": "fixture-cli",
        "k": 2,
        "execution_scope": "FIXTURE_ONLY",
        "max_source_rows": 99,
        "seed": 7,
        "num_boost_round": 20,
        "max_depth": 4,
        "learning_rate": 0.2,
        "min_child_weight": 0.0,
        "l2": 2.5,
        "early_stopping_rounds": 5,
    }


def test_xgboost_seed_is_required_before_source_loading(cli_case, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing seed must fail before loading a source")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    assert offline_cli.main([*train_arguments(cli_case), "--algorithm", "xgboost"]) == 2
    assert '"status": "FAILED"' in capsys.readouterr().out
    assert cli_case.selected_modules == []


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--seed", "1"),
        ("--num-boost-round", "10"),
        ("--max-depth", "3"),
        ("--learning-rate", "0.1"),
        ("--min-child-weight", "1"),
        ("--l2", "1"),
        ("--early-stopping-rounds", "5"),
    ],
)
@pytest.mark.parametrize("command", ["prepare", "verify", "train", "verify-model"])
def test_xgboost_options_cannot_be_ignored_by_other_contexts(
    cli_case, monkeypatch, capsys, command, flag, value
):
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong-context options must fail before source loading")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    assert offline_cli.main([command, *cli_case.common, flag, value]) == 2
    assert "XGBoost options require train --algorithm xgboost" in capsys.readouterr().out
    assert cli_case.selected_modules == []


@pytest.mark.parametrize("command", ["prepare", "verify", "verify-model"])
@pytest.mark.parametrize("algorithm", ["numpy-logistic", "xgboost"])
def test_algorithm_selector_is_only_valid_for_train(
    cli_case, monkeypatch, command, algorithm, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong-command algorithm selection must fail before source loading")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    assert offline_cli.main([command, *cli_case.common, "--algorithm", algorithm]) == 2
    assert "--algorithm is only valid for train" in capsys.readouterr().out
    assert cli_case.selected_modules == []


@pytest.mark.parametrize(
    "kind,algorithm",
    [
        ("NUMPY_LOGISTIC_PROXY_BASELINE", "numpy-logistic"),
        ("XGBOOST_BINARY_LOGISTIC_PROXY", "xgboost"),
    ],
)
def test_verify_model_routes_only_from_saved_kind(cli_case, policy, kind, algorithm):
    write_json(cli_case.output / "model.json", {"model_kind": kind})
    assert (
        offline_cli.main(
            [
                "verify-model",
                *cli_case.common,
                "--dataset",
                str(cli_case.dataset),
                "--model",
                str(cli_case.output),
                "--max-source-rows",
                "99",
            ]
        )
        == 0
    )
    assert cli_case.selected_modules == [algorithm]
    name, args, kwargs = cli_case.calls[0]
    assert name == "verify-model"
    assert args == (cli_case.output, cli_case.dataset, cli_case.source, policy)
    assert kwargs == {"execution_scope": "FIXTURE_ONLY", "max_source_rows": 99}


@pytest.mark.parametrize("metadata", [{}, {"model_kind": None}, {"model_kind": "UNKNOWN"}])
def test_unknown_saved_model_kind_never_falls_back_to_numpy(cli_case, metadata, capsys):
    write_json(cli_case.output / "model.json", metadata)
    assert (
        offline_cli.main(
            [
                "verify-model",
                *cli_case.common,
                "--dataset",
                str(cli_case.dataset),
                "--model",
                str(cli_case.output),
            ]
        )
        == 2
    )
    assert "Missing or unsupported saved offline model kind" in capsys.readouterr().out
    assert cli_case.selected_modules == []


def test_pending_policy_precedes_source_loading_and_training_import(cli_case, monkeypatch):
    write_json(cli_case.policy_path, asdict(R02Policy()))

    def forbidden(*args, **kwargs):
        pytest.fail("Pending policy must fail before source loading or training import")

    monkeypatch.setattr(offline_cli, "load_index", forbidden)
    monkeypatch.setattr(offline_cli, "_training_module", forbidden)
    assert (
        offline_cli.main([*train_arguments(cli_case), "--algorithm", "xgboost", "--seed", "1"]) == 2
    )
    assert not cli_case.output.exists()


@pytest.mark.parametrize(
    "error",
    [
        "XGBoost import unavailable: install the optional training dependency",
        "seed must be an integer in [0, 2147483647]",
        "early_stopping_rounds must not exceed num_boost_round",
    ],
)
def test_adapter_validation_and_dependency_errors_are_failed_json(
    cli_case, monkeypatch, capsys, error
):
    def rejected(*args, **kwargs):
        raise ValueError(error)

    monkeypatch.setattr(
        offline_cli,
        "_training_module",
        lambda algorithm: SimpleNamespace(train_evaluate=rejected),
    )
    assert (
        offline_cli.main([*train_arguments(cli_case), "--algorithm", "xgboost", "--seed", "1"]) == 2
    )
    assert json.loads(capsys.readouterr().out) == {"status": "FAILED", "error": error}
    assert not cli_case.output.exists()


def test_unrelated_internal_import_error_is_not_relabelled_as_missing_xgboost(
    cli_case, monkeypatch, capsys
):
    def broken(*args, **kwargs):
        raise ImportError("unrelated adapter implementation error")

    monkeypatch.setattr(
        offline_cli,
        "_training_module",
        lambda algorithm: SimpleNamespace(train_evaluate=broken),
    )
    assert (
        offline_cli.main([*train_arguments(cli_case), "--algorithm", "xgboost", "--seed", "1"]) == 2
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "FAILED",
        "error": "unrelated adapter implementation error",
    }
    assert not cli_case.output.exists()


@pytest.mark.parametrize("command", ["prepare", "verify"])
def test_nontraining_commands_do_not_resolve_a_training_module(cli_case, monkeypatch, command):
    def forbidden(*args, **kwargs):
        pytest.fail("Preparation and dataset verification must not import training adapters")

    monkeypatch.setattr(offline_cli, "_training_module", forbidden)
    monkeypatch.setattr(
        offline_cli, "build_dataset", lambda *args, **kwargs: {"status": "VERIFIED"}
    )
    monkeypatch.setattr(
        offline_cli, "verify_dataset", lambda *args, **kwargs: {"status": "VERIFIED"}
    )
    options = (
        [
            "--output",
            str(cli_case.dataset),
            "--train-before",
            "2022-01-02",
            "--test-from",
            "2022-01-04",
        ]
        if command == "prepare"
        else ["--dataset", str(cli_case.dataset)]
    )
    assert offline_cli.main([command, *cli_case.common, *options]) == 0
