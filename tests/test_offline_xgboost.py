"""소규모 네이티브 XGBoost 저장·읽기와 오프라인 선택·무결성 경계를 검사합니다."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import _build_review_dataset, synthetic_support

from aml_ml import offline_pipeline, offline_training, offline_xgboost
from aml_ml.offline_pipeline import FEATURE_NAMES
from aml_ml.offline_rules import fingerprint
from aml_ml.storage import read_json, sha256, write_json

SCOPE = "FIXTURE_ONLY"


def _train(dataset, source, policy, output, **overrides):
    options = {
        "execution_scope": SCOPE,
        "model_version": "synthetic-xgboost-review",
        "k": 1,
        "seed": 17,
        "num_boost_round": 4,
        "max_depth": 2,
        # 경보가 두 개인 합성 train 분할에서만 이 작은 리프 제한을 사용합니다.
        "min_child_weight": 0.0,
        "early_stopping_rounds": 2,
        "support_policy": synthetic_support("XGBOOST_BINARY_LOGISTIC_PROXY"),
    }
    options.update(overrides)
    return offline_xgboost.train_evaluate(dataset, source, policy, output, **options)


def _native_model(folder):
    return offline_xgboost._xgboost().Booster(
        params={"device": "cpu", "nthread": 1}, model_file=folder / "booster.ubj"
    )


def _resign_metadata(folder, state):
    write_json(folder / "model.json", state)
    manifest = read_json(folder / "manifest.json")
    manifest["files"]["model.json"] = sha256(folder / "model.json")
    manifest["model_fingerprint"] = fingerprint(manifest["files"])
    write_json(folder / "manifest.json", manifest)


def _replace_source_labels(source, rows):
    path = source.part_path("labels", 0)
    pq.write_table(pa.Table.from_pylist(rows), path)
    source.labels["parts"][0]["files"][0]["sha256"] = sha256(path)
    write_json(path.parent / "manifest.json", source.labels)


@pytest.fixture
def xgb_dataset(source, policy, tmp_path):
    dataset = tmp_path / "xgb-data"
    _build_review_dataset(source, policy, dataset)
    return dataset


@pytest.fixture
def trained_xgb(xgb_dataset, source, policy, tmp_path):
    model = tmp_path / "xgb-model"
    result = _train(xgb_dataset, source, policy, model)
    return xgb_dataset, model, result


def test_native_fit_save_reload_and_explicit_verify(trained_xgb, source, policy):
    dataset, model, result = trained_xgb
    state = offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)
    booster = _native_model(model)
    xgb = offline_xgboost._xgboost()
    assert result["status"] == "VERIFIED"
    assert result["deployed"] is False
    assert result["service_id_mapping_verified"] is False
    assert state["model_kind"] == "XGBOOST_BINARY_LOGISTIC_PROXY"
    assert state["schema_version"] == "offline-xgboost-v1"
    assert state["parameters"]["seed"] == 17
    assert state["fit_scope"] == "TRAIN_COMPLETE_PROXY_ONLY"
    assert state["pr_auc_definition"] == "AVERAGE_PRECISION_STEP_INTEGRAL"
    assert result["pr_auc_definition"] == state["pr_auc_definition"]
    assert state["validated_dataset_manifest_sha256"] == sha256(dataset / "manifest.json")
    assert state["selected_rounds"] == state["best_iteration"] + 1
    assert booster.num_boosted_rounds() == state["selected_rounds"]
    assert booster.feature_names == state["feature_columns"]
    for key, expected in (
        ("device", "cpu"),
        ("tree_method", "hist"),
        ("nthread", 1),
        ("objective", "binary:logistic"),
    ):
        assert state["native_parameters"][key] == expected
    assert set(state["training_history"]) == {"validation"}
    assert set(state["training_history"]["validation"]) == {"logloss"}
    assert {p.name for p in model.iterdir()} == {
        "booster.ubj",
        "model.json",
        "preprocessing.json",
        "evaluation.json",
        "predictions.json",
        "manifest.json",
    }
    stored = read_json(model / "predictions.json")
    for split in ("train", "validation", "test"):
        x, _, _, _ = offline_training._labelled(dataset, split, state["feature_columns"])
        native = booster.predict(
            xgb.DMatrix(x, feature_names=state["feature_columns"], nthread=1),
            iteration_range=(0, state["selected_rounds"]),
            validate_features=True,
        )
        assert native.shape == (len(x),)
        assert np.isfinite(native).all()
        assert ((native >= 0) & (native <= 1)).all()
        np.testing.assert_array_equal(native, stored[split])
    assert (
        offline_xgboost.verify_model(model, dataset, source, policy, execution_scope=SCOPE)
        == result
    )


def test_same_seed_repeats_tree_scores_and_selection_in_same_environment(
    trained_xgb, source, policy, tmp_path
):
    dataset, first, _ = trained_xgb
    second = tmp_path / "repeat-model"
    _train(dataset, source, policy, second, model_version="synthetic-xgboost-repeat")
    left, right = read_json(first / "model.json"), read_json(second / "model.json")
    assert _native_model(first).get_dump(dump_format="json") == _native_model(second).get_dump(
        dump_format="json"
    )
    for key in (
        "parameters",
        "native_parameters",
        "environment_versions",
        "selected_rounds",
        "best_iteration",
        "recommended_threshold",
        "training_history",
        "preprocessing_sha256",
    ):
        assert left[key] == right[key]
    assert read_json(first / "predictions.json") == read_json(second / "predictions.json")
    assert read_json(first / "evaluation.json") == read_json(second / "evaluation.json")


def test_early_stopping_saves_best_rounds_instead_of_last_plateau_model(
    xgb_dataset, source, policy, tmp_path
):
    output = tmp_path / "plateau-model"
    # 이 최소 자식 가중치 제한에서는 균형 잡힌 train 경보 두 개를 분할할 수 없습니다.
    _train(
        xgb_dataset,
        source,
        policy,
        output,
        min_child_weight=1.0,
        num_boost_round=8,
        early_stopping_rounds=2,
    )
    state = read_json(output / "model.json")
    assert state["best_iteration"] == 0
    assert state["selected_rounds"] == 1
    assert state["selected_rounds"] < state["training_rounds"] < 8
    assert _native_model(output).num_boosted_rounds() == 1
    for scores in read_json(output / "predictions.json").values():
        np.testing.assert_array_equal(scores, np.full(len(scores), 0.5))


def test_validation_only_early_stopping_and_test_after_threshold_freeze(
    xgb_dataset, source, policy, tmp_path, monkeypatch
):
    columns = read_json(xgb_dataset / "manifest.json")["feature_columns"]
    _, train_y, _, _ = offline_xgboost._labelled(xgb_dataset, "train", columns)
    _, valid_y, _, _ = offline_xgboost._labelled(xgb_dataset, "validation", columns)
    # 날짜마다 UID 순서가 다를 수 있으므로 배열이 일치할 때만 라벨을 바꿉니다.
    if np.array_equal(train_y, valid_y):
        labels = pq.read_table(source.part_path("labels", 0)).to_pylist()
        for row in labels[5:8]:
            row["source_positive"] = 0
        for row in labels[8:10]:
            row["source_positive"] = 1
        _replace_source_labels(source, labels)
        xgb_dataset = tmp_path / "opposite-validation-data"
        _build_review_dataset(source, policy, xgb_dataset)
    events, replays = [], []
    original_labelled = offline_xgboost._labelled
    original_selector = offline_xgboost.select_validation_threshold
    original_verify = offline_training.verify_dataset
    xgb = offline_xgboost._xgboost()
    original_train = xgb.train
    columns = read_json(xgb_dataset / "manifest.json")["feature_columns"]
    _, expected_train_y, _, _ = original_labelled(xgb_dataset, "train", columns)
    _, expected_valid_y, _, _ = original_labelled(xgb_dataset, "validation", columns)
    assert not np.array_equal(expected_train_y, expected_valid_y)

    def tracked_labelled(folder, split, features):
        events.append(split)
        return original_labelled(folder, split, features)

    def tracked_selector(y, scores):
        assert "booster_frozen" in events
        assert "test" not in events
        result = original_selector(y, scores)
        events.append("threshold_frozen")
        return result

    def tracked_train(params, dtrain, *args, **kwargs):
        assert "test" not in events
        np.testing.assert_array_equal(dtrain.get_label(), expected_train_y)
        watchlist = kwargs["evals"]
        assert len(watchlist) == 1 and watchlist[0][1] == "validation"
        assert watchlist[0][0] is not dtrain
        np.testing.assert_array_equal(watchlist[0][0].get_label(), expected_valid_y)
        stopping = kwargs["callbacks"][0]
        assert stopping.data == "validation"
        assert stopping.metric_name == "logloss"
        assert stopping.save_best is True
        assert stopping.maximize is False
        events.append("native_fit")
        booster = original_train(params, dtrain, *args, **kwargs)
        events.append("booster_frozen")
        return booster

    def tracked_verify(*args, **kwargs):
        replays.append("replay")
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(offline_xgboost, "_labelled", tracked_labelled)
    monkeypatch.setattr(offline_xgboost, "select_validation_threshold", tracked_selector)
    monkeypatch.setattr(offline_training, "verify_dataset", tracked_verify)
    monkeypatch.setattr(xgb, "train", tracked_train)
    output = tmp_path / "selection-model"
    result = _train(xgb_dataset, source, policy, output)
    assert replays == ["replay"]
    assert events.index("test") > events.index("threshold_frozen")
    assert events.count("native_fit") == 1
    assert (
        offline_xgboost.verify_model(output, xgb_dataset, source, policy, execution_scope=SCOPE)
        == result
    )
    assert replays == ["replay", "replay"]


def test_heldout_test_labels_cannot_select_tree_rounds_threshold_or_preprocessing(
    trained_xgb, source, policy, tmp_path
):
    dataset, first, _ = trained_xgb
    original = read_json(first / "model.json")
    original_trees = _native_model(first).get_dump(dump_format="json")
    original_scores = read_json(first / "predictions.json")
    labels = pq.read_table(source.part_path("labels", 0)).to_pylist()
    for row in labels[10:]:
        row["source_positive"] = int(row["source_row_id"] >= 13)
    _replace_source_labels(source, labels)
    second_data, second = tmp_path / "changed-test-data", tmp_path / "changed-test-model"
    _build_review_dataset(source, policy, second_data)
    _train(second_data, source, policy, second, model_version="synthetic-changed-test-labels")
    changed = read_json(second / "model.json")
    assert original_trees == _native_model(second).get_dump(dump_format="json")
    for key in (
        "parameters",
        "selected_rounds",
        "best_iteration",
        "recommended_threshold",
        "training_history",
        "feature_columns",
        "preprocessing_sha256",
    ):
        assert original[key] == changed[key]
    assert original_scores == read_json(second / "predictions.json")
    assert (first / "preprocessing.json").read_bytes() == (
        second / "preprocessing.json"
    ).read_bytes()
    assert original["dataset_fingerprint"] != changed["dataset_fingerprint"]
    assert sha256(dataset / "y_test.parquet") != sha256(second_data / "y_test.parquet")


def test_null_test_truth_preserved_but_excluded_and_insufficient_support_blocks_fit(
    source, policy, tmp_path
):
    labels = pq.read_table(source.part_path("labels", 0)).to_pylist()
    labels[13]["source_positive"] = None
    _replace_source_labels(source, labels)
    dataset, output = tmp_path / "nullable-data", tmp_path / "nullable-model"
    _build_review_dataset(source, policy, dataset)
    with pytest.raises(ValueError, match="NOT_READY"):
        _train(dataset, source, policy, output)
    ledger = read_json(dataset / "ledger.json")
    unknown = [item for item in ledger if item["y_alert_proxy"] is None]
    assert len(unknown) == 1 and not unknown[0]["supervised_eligible"]
    assert pq.read_table(dataset / "y_test.parquet").num_rows == 1
    assert not output.exists()


@pytest.mark.parametrize("unknown_index", [3, 8])
def test_incomplete_train_or_validation_class_is_rejected(source, policy, tmp_path, unknown_index):
    labels = pq.read_table(source.part_path("labels", 0)).to_pylist()
    labels[unknown_index]["source_positive"] = None
    _replace_source_labels(source, labels)
    dataset, output = tmp_path / "one-class-data", tmp_path / "one-class-model"
    with pytest.raises(ValueError):
        _build_review_dataset(source, policy, dataset)
        _train(dataset, source, policy, output)
    assert not output.exists()


def test_tied_scores_keep_uid_top_k_and_step_average_precision(
    xgb_dataset, source, policy, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        offline_xgboost,
        "_scores",
        lambda booster, x, columns, selected_rounds: np.full(len(x), 0.5),
    )
    output = tmp_path / "tied-model"
    result = _train(xgb_dataset, source, policy, output)
    for split in ("train", "validation", "test"):
        truth = pq.read_table(xgb_dataset / f"y_{split}.parquet").to_pylist()
        first = min(truth, key=lambda row: row["training_alert_uid"])
        evaluation = result["evaluation"][split]
        assert evaluation["top_k_tie_policy"] == "SCORE_DESC_THEN_TRAINING_ALERT_UID_ASC"
        assert evaluation["precision_at_k"] == first["y_alert_proxy"]
        assert evaluation["average_precision"] == 0.5
        assert evaluation["pr_auc"] == evaluation["average_precision"]
        assert evaluation["pr_auc_definition"] == "AVERAGE_PRECISION_STEP_INTEGRAL"


@pytest.mark.parametrize("scores", [[np.nan], [np.inf], [-0.01], [1.01], [[0.5]], []])
def test_invalid_native_probability_output_is_rejected(scores):
    class InvalidBooster:
        def predict(self, data, **kwargs):
            return np.asarray(scores)

    with pytest.raises(ValueError, match="invalid binary logistic scores"):
        offline_xgboost._scores(InvalidBooster(), np.array([[1.0]]), ["feature"], 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_source_rows": 1},
        {"execution_scope": "CONFIRMED"},
        {"seed": True},
        {"seed": -1},
        {"seed": 2**31},
        {"num_boost_round": 0},
        {"early_stopping_rounds": 5},
        {"max_depth": 17},
        {"learning_rate": np.nan},
        {"min_child_weight": -1},
        {"l2": np.inf},
        {"k": True},
    ],
)
def test_training_guard_precedes_backend_import_and_output(
    source, policy, tmp_path, monkeypatch, overrides
):
    def backend_forbidden():
        pytest.fail("Rejected requests must not import the native model backend")

    monkeypatch.setattr(offline_xgboost, "_xgboost", backend_forbidden)
    output = tmp_path / "forbidden-model"
    with pytest.raises(ValueError):
        _train(tmp_path / "missing-data", source, policy, output, **overrides)
    assert not output.exists()


def test_existing_model_directory_is_preserved(trained_xgb, source, policy):
    dataset, model, _ = trained_xgb
    before = {path.name: sha256(path) for path in model.iterdir()}
    with pytest.raises(FileExistsError):
        _train(dataset, source, policy, model)
    assert {path.name: sha256(path) for path in model.iterdir()} == before


def test_load_predict_do_not_replay_data_or_refit(trained_xgb, source, policy, monkeypatch):
    dataset, model, _ = trained_xgb

    def forbidden(*args, **kwargs):
        pytest.fail("Loading and prediction must not replay source data or refit preprocessing")

    monkeypatch.setattr(offline_training, "verify_dataset", forbidden)
    monkeypatch.setattr(offline_pipeline, "_derive", forbidden)
    monkeypatch.setattr(offline_pipeline.AlertPreprocessor, "partial_fit", forbidden)
    offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)
    raw = pq.read_table(dataset / "raw.parquet")
    scores = offline_xgboost.predict_raw(model, raw, dataset, source, policy, execution_scope=SCOPE)
    assert scores.shape == (len(raw),)
    assert np.isfinite(scores).all()
    for invalid in (
        raw.select(list(reversed(FEATURE_NAMES))),
        raw.append_column("training_alert_uid", pa.array(["forbidden"] * len(raw))),
        raw.append_column("y_alert_proxy", pa.array([1] * len(raw))),
    ):
        with pytest.raises(ValueError):
            offline_xgboost.predict_raw(
                model, invalid, dataset, source, policy, execution_scope=SCOPE
            )


@pytest.mark.parametrize(
    "field",
    [
        "environment_versions",
        "code_sha256",
        "feature_columns",
        "selected_rounds",
        "validated_dataset_manifest_sha256",
        "recommended_threshold",
        "native_parameters",
    ],
)
def test_resigned_metadata_mutation_is_rejected(trained_xgb, source, policy, field):
    dataset, model, _ = trained_xgb
    state = read_json(model / "model.json")
    if field in {"environment_versions", "code_sha256"}:
        state[field][next(iter(state[field]))] = "changed"
    elif field == "feature_columns":
        state[field] = list(reversed(state[field]))
    elif field == "selected_rounds":
        state[field] += 1
    elif field == "validated_dataset_manifest_sha256":
        state[field] = "0" * 64
    elif field == "recommended_threshold":
        state[field] = 0.75 if state[field] != 0.75 else 0.25
    else:
        state[field]["objective"] = "reg:squarederror"
    _resign_metadata(model, state)
    with pytest.raises(ValueError):
        offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("helper", ["_environment_versions", "_code_hashes"])
def test_changed_observed_environment_or_code_is_rejected(
    trained_xgb, source, policy, monkeypatch, helper
):
    dataset, model, _ = trained_xgb
    original = getattr(offline_xgboost, helper)

    def changed():
        values = dict(original())
        values[next(iter(values))] = "changed-observed-value"
        return values

    monkeypatch.setattr(offline_xgboost, helper, changed)
    with pytest.raises(ValueError):
        offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("mutation", ["extra-file", "extra-directory", "missing", "hash"])
def test_exact_inventory_and_native_model_hash_are_enforced(trained_xgb, source, policy, mutation):
    dataset, model, _ = trained_xgb
    if mutation == "extra-file":
        (model / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "extra-directory":
        (model / "unexpected").mkdir()
    elif mutation == "missing":
        (model / "booster.ubj").unlink()
    else:
        native = model / "booster.ubj"
        native.write_bytes(native.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)


@pytest.mark.parametrize("target", ["dataset-manifest", "source-part", "preprocessing"])
def test_stale_dataset_source_or_preprocessing_is_rejected(trained_xgb, source, policy, target):
    dataset, model, _ = trained_xgb
    if target == "dataset-manifest":
        path = dataset / "manifest.json"
    elif target == "source-part":
        path = source.part_path("transactions", 0)
    else:
        path = model / "preprocessing.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        offline_xgboost.load_model(model, dataset, source, policy, execution_scope=SCOPE)


def test_manifest_change_during_deep_verification_prevents_training_output(
    xgb_dataset, source, policy, tmp_path, monkeypatch
):
    original = offline_training.verify_dataset

    def changed_after_verify(*args, **kwargs):
        result = original(*args, **kwargs)
        manifest = xgb_dataset / "manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(offline_training, "verify_dataset", changed_after_verify)
    output = tmp_path / "changed-snapshot-model"
    with pytest.raises(ValueError, match="changed during"):
        _train(xgb_dataset, source, policy, output)
    assert not output.exists()


def test_legacy_logistic_artifacts_keep_their_existing_loader(
    xgb_dataset, source, policy, tmp_path
):
    output = tmp_path / "legacy-logistic-model"
    result = offline_training.train_evaluate(
        xgb_dataset,
        source,
        policy,
        output,
        execution_scope=SCOPE,
        model_version="synthetic-logistic-compatibility",
        support_policy=synthetic_support(),
        k=1,
        iterations=3,
    )
    state = offline_training.load_model(output, xgb_dataset, source, policy, execution_scope=SCOPE)
    assert state["model_kind"] == "NUMPY_LOGISTIC_PROXY_BASELINE"
    assert not (output / "booster.ubj").exists()
    assert (
        offline_training.verify_model(output, xgb_dataset, source, policy, execution_scope=SCOPE)
        == result
    )


def test_code_inventory_includes_adapter_and_shared_training_helpers(trained_xgb):
    _, model, _ = trained_xgb
    state = read_json(model / "model.json")
    assert {"offline_xgboost.py", "offline_training.py"} <= set(state["code_sha256"])
    for name in ("offline_xgboost.py", "offline_training.py"):
        assert state["code_sha256"][name] == sha256(Path(offline_training.__file__).with_name(name))
