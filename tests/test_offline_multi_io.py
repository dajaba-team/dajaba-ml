"""Small I/O contract tests. No original dataset or production run is opened."""

import pytest

from aml_ml.offline_multi import load_tables as legacy_load_tables
from aml_ml.offline_multi_io import append, load_tables, query_batches


def test_existing_source_loader_import_remains_compatible():
    assert legacy_load_tables is load_tables


def test_append_unregisters_input_after_query_failure():
    calls = []

    class Connection:
        def register(self, name, table):
            calls.append(("register", name, table))

        def execute(self, query):
            raise RuntimeError("synthetic write error")

        def unregister(self, name):
            calls.append(("unregister", name))

    table = object()
    with pytest.raises(RuntimeError, match="synthetic write error"):
        append(Connection(), "features", table)
    assert calls == [("register", "_batch_input", table), ("unregister", "_batch_input")]


def test_batch_reader_closes_child_cursor_when_consumer_stops():
    calls = []

    class Cursor:
        def execute(self, query):
            calls.append(query)
            return self

        def fetch_record_batch(self, size):
            calls.append(size)
            return iter(["first", "second"])

        def close(self):
            calls.append("closed")

    class Connection:
        def cursor(self):
            return Cursor()

    batches = query_batches(Connection(), "SELECT synthetic", size=7)
    assert next(batches) == "first"
    batches.close()
    assert calls == ["SELECT synthetic", 7, "closed"]
