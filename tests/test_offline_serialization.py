"""세 형식의 타입·원본 금액·유효한 JSON 배열을 독립적으로 대조하는 회귀입니다."""

import json
from datetime import datetime
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from offline_test_data import prepare

from aml_ml import offline_serialization as codec
from aml_ml.offline_contract import DATASET_FILES


def _table():
    schema = pa.schema(
        [
            ("uid", pa.string()),
            ("amount", pa.decimal256(40, 8)),
            ("currency", pa.string()),
            ("ts", pa.timestamp("us")),
            ("truth", pa.int8()),
            ("flag", pa.bool_()),
            ("score", pa.float64()),
            ("refs", pa.list_(pa.string())),
            ("paths", pa.list_(pa.list_(pa.string()))),
        ]
    )
    return pa.Table.from_pylist(
        [
            dict(
                uid='한글,"line\n\\N',
                amount=Decimal("123456789012345678901234.12345678"),
                currency="원본 통화",
                ts=datetime(2022, 1, 1, 12, 1, 2, 345678),
                truth=None,
                flag=False,
                score=-0.0,
                refs=["00001", "\\N", ""],
                paths=[["a", "b"], []],
            ),
            dict(
                uid="0000002",
                amount=Decimal("0.00000001"),
                currency="USD",
                ts=None,
                truth=1,
                flag=True,
                score=1.25,
                refs=[],
                paths=None,
            ),
            dict(
                uid="\\N",
                amount=None,
                currency="",
                ts=None,
                truth=0,
                flag=None,
                score=None,
                refs=None,
                paths=[],
            ),
        ],
        schema=schema,
    )


def test_exact_decimal_unicode_null_escape_nested_order_and_empty_arrays(tmp_path):
    table = _table()
    for name, selected in (("values", table), ("empty", table.slice(0, 0))):
        codec.write_three(selected, tmp_path / f"{name}.parquet")
        assert codec.verify_table(tmp_path, name)["rows"] == len(selected)
        array = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        assert isinstance(array, list)
        if name == "values":
            assert array[0]["amount"] == "123456789012345678901234.12345678"
            assert array[0]["truth"] is None and array[0]["refs"][0] == "00001"
            assert array[2]["uid"] == "\\N" and array[2]["currency"] == ""
        else:
            assert array == []


@pytest.mark.parametrize(
    "text",
    [
        '{"a":1}',
        '{"a":1}\n{"a":2}',
        '[{"a":1},]',
        '[{"a":1}',
        '[{"a":1,"a":2}]',
        '[{"a":1}] trailing',
        "[null]",
        '[{"a":NaN}]',
        '[{"a":Infinity}]',
        '[{"a":-Infinity}]',
    ],
)
def test_object_jsonl_partial_duplicate_and_trailing_data_are_rejected(tmp_path, text):
    path = tmp_path / "invalid.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        list(codec.iter_json_array(path, max_record_bytes=4096))


@pytest.mark.parametrize("tamper", ["money_float", "order", "uid", "extra", "missing", "bool"])
def test_readback_rejects_type_value_order_and_cardinality_changes(tmp_path, tamper):
    codec.write_three(_table(), tmp_path / "data.parquet")
    path = tmp_path / "data.json"
    array = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "money_float":
        array[0]["amount"] = 1.23
    elif tamper == "order":
        array.reverse()
    elif tamper == "uid":
        array[1]["uid"] = 2
    elif tamper == "extra":
        array.append(array[-1])
    elif tamper == "missing":
        array.pop()
    else:
        array[0]["flag"] = 0
    path.write_text(json.dumps(array, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError):
        codec.verify_table(tmp_path, "data")


@pytest.mark.parametrize("form", ["json", "csv", "parquet"])
def test_column_order_changes_are_rejected_in_each_representation(tmp_path, form):
    table = _table()
    codec.write_three(table, tmp_path / "data.parquet")
    path = tmp_path / f"data.{form}"
    if form == "json":
        array = json.loads(path.read_text(encoding="utf-8"))
        array[0] = {key: array[0][key] for key in reversed(table.schema.names)}
        path.write_text(json.dumps(array, ensure_ascii=False), encoding="utf-8")
        message = "column names/order"
    elif form == "csv":
        content = path.read_text(encoding="utf-8")
        _, remaining = content.split("\n", 1)
        path.write_text(",".join(reversed(table.schema.names)) + "\n" + remaining, encoding="utf-8")
        message = "header/order"
    else:
        pq.write_table(table.select(list(reversed(table.schema.names))), path)
        message = "Parquet schema differs"
    with pytest.raises(ValueError, match=message):
        codec.verify_table(tmp_path, "data")


@pytest.mark.parametrize("value", ["2022-01-01T00:00:00+09:00", "2022-01-01T00:00:00.1234567"])
def test_timestamp_timezone_and_excess_precision_are_rejected(value):
    with pytest.raises(ValueError, match="naive|Noncanonical or over-precision"):
        codec.typed_value(value, pa.field("ts", pa.timestamp("us")))


def test_completed_dataset_has_eleven_tables_and_all_three_formats(source, policy, tmp_path):
    output = tmp_path / "dataset"
    assert prepare(source, policy, output)["status"] == "VERIFIED"
    assert set(codec.TABLES) == {
        "alerts",
        "targets",
        "ledger",
        "evidence",
        "raw",
        "X_train",
        "y_train",
        "X_validation",
        "y_validation",
        "X_test",
        "y_test",
    }
    assert {path.name for path in output.iterdir()} == DATASET_FILES | {"manifest.json"}
    codec.verify_formats(output)
    for name in codec.TABLES:
        array = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        assert len(array) == pq.read_metadata(output / f"{name}.parquet").num_rows


def test_nested_workset_overflow_never_calls_to_pylist_and_preserves_primary(tmp_path, monkeypatch):
    table = _table()
    path = tmp_path / "large.parquet"

    class Batch:
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def slice(self, start, length=None):
            return Batch(self.size - start if length is None else length)

        def to_pylist(self):
            pytest.fail("to_pylist was called before the recursive budget rejection")

    class Table:
        schema = table.schema
        nbytes = table.nbytes

        def to_batches(self, **kwargs):
            return [Batch(len(table))]

    monkeypatch.setattr(codec, "python_decode_bytes", lambda chunk: 1024**3)
    with pytest.raises(ValueError, match="workset"):
        with codec.ThreeWriter(path, table.schema, max_bytes=1024**2) as writer:
            writer.write_table(Table())
    # 미완료 배열은 실패 기록으로 보존하고 완성 결과로 게시하지 않습니다.
    with pytest.raises(ValueError):
        list(codec.iter_json_array(path.with_suffix(".json")))
    assert not (tmp_path / "formats.json").exists()


def test_writer_cleanup_does_not_replace_original_error(tmp_path, monkeypatch):
    table = _table()
    writer = codec.ThreeWriter(tmp_path / "data.parquet", table.schema)
    writer.__enter__()
    original = writer.stack.close

    def broken_close():
        original()
        raise OSError("close failed")

    monkeypatch.setattr(writer.stack, "close", broken_close)
    primary = KeyboardInterrupt("original")
    writer.__exit__(KeyboardInterrupt, primary, None)
    assert any("close" in note for note in primary.__notes__)
