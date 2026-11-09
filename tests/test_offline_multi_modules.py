"""Compatibility checks for responsibility-only multi-rule module extraction."""

import pytest

from aml_ml import (
    offline_multi,
    offline_multi_dataset,
    offline_multi_metadata,
    offline_multi_recovery,
)


@pytest.mark.parametrize(
    ("name", "module"),
    [
        ("candidate_metadata", offline_multi_metadata),
        ("META_SCHEMA", offline_multi_metadata),
        ("FACT_SCHEMA", offline_multi_metadata),
        ("feature_table", offline_multi_dataset),
        ("export_dataset", offline_multi_dataset),
        ("RAW_SCHEMA", offline_multi_dataset),
        ("restore_candidate_prefix", offline_multi_recovery),
    ],
)
def test_previous_imports_resolve_to_same_implementation(name, module):
    assert getattr(offline_multi, name) is getattr(module, name)
