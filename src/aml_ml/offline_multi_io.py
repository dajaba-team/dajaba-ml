"""Bounded SQL/Arrow I/O and immutable source checks for multi-rule preparation.

Rule detection, split policy and training stay outside this module. Callers own
connections; readers close their child cursors and writers close all formats.
"""

from contextlib import ExitStack

import pyarrow as pa

from .offline_multi_progress import log
from .offline_serialization import ThreeWriter

BATCH = 1024


def append(con, name, table, *, create=False):
    con.register("_batch_input", table)
    try:
        if create:
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM _batch_input")
        else:
            con.execute(f"INSERT INTO {name} SELECT * FROM _batch_input")
    finally:
        con.unregister("_batch_input")


def query_batches(con, query, size=BATCH):
    cursor = con.cursor()
    try:
        reader = cursor.execute(query).fetch_record_batch(size)
        yield from reader
    finally:
        cursor.close()


def export_query(con, query, path, schema=None):
    with ExitStack() as stack:
        writer = None
        for batch in query_batches(con, query):
            table = pa.Table.from_batches([batch])
            if schema is not None:
                table = table.cast(schema)
            if writer is None:
                writer = stack.enter_context(ThreeWriter(path, table.schema))
            writer.write_table(table)
        if writer is None:
            if schema is None:
                schema = con.execute(query + " LIMIT 0").fetch_arrow_table().schema
            stack.enter_context(ThreeWriter(path, schema))


def load_tables(con, source, folder):
    log(folder, "verify_source_parts", source_id=source.id)
    paths = {}
    for which in ("transactions", "labels"):
        paths[which] = [str(source.part_path(which, i)) for i in range(len(source.parts(which)))]
    con.from_parquet(paths["transactions"]).create_view("input_transactions")
    con.from_parquet(paths["labels"]).create_view("input_labels")
    extras = (
        (", NULL::DECIMAL(38,12) AS amount_paid, NULL::DECIMAL(38,12) AS amount_received")
        if source.kind != "ibm"
        else ""
    )
    con.execute("CREATE TABLE tx AS SELECT *" + extras + " FROM input_transactions")
    con.execute("CREATE TABLE labels AS SELECT * FROM input_labels")
    count = source.metadata["input_rows"]
    for table in ("tx", "labels"):
        got = con.execute(
            f"""
            SELECT count(*), count(DISTINCT source_row_id),
                   min(source_row_id), max(source_row_id),
                   count(*) FILTER (WHERE transaction_uid IS DISTINCT FROM ? || ':' || source_row_id)
            FROM {table}
        """,
            [source.metadata["source_release"]],
        ).fetchone()
        if got != (count, count, 0, count - 1, 0):
            raise ValueError(f"Full-source identity/count mismatch: {table}: {got}")
    bad = con.execute("""
        SELECT count(*) FROM tx
        WHERE ts IS NULL OR from_account_key IS NULL OR to_account_key IS NULL
        OR from_account_key = '' OR to_account_key = '' OR is_self_transfer IS NULL
    """).fetchone()[0]
    if bad:
        raise ValueError("Missing transaction identity/time/self-transfer fields")
    if con.execute("""SELECT count(*) FROM labels
                      WHERE source_positive NOT IN (0,1)""").fetchone()[0]:
        raise ValueError("Source label must be 0, 1 or null")
    log(folder, "source_loaded", input_rows=count)
