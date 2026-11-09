"""Build causal account features and export train-fitted multi-rule datasets."""

import pyarrow as pa
import pyarrow.parquet as pq

from .history import Event, Window
from .offline_account_stream import _features
from .offline_features import FEATURE_NAMES, AlertPreprocessor
from .offline_multi_io import BATCH, append, export_query, query_batches
from .offline_multi_progress import log
from .offline_serialization import ThreeWriter, finish_formats, verify_table
from .schema import currency_bucket, epoch_us
from .storage import write_json

RAW_SCHEMA = pa.schema([(name, pa.float64()) for name in FEATURE_NAMES])


def feature_table(con, source, folder):
    feature_schema = pa.schema([("training_alert_uid", pa.string())] + list(RAW_SCHEMA))
    append(con, "features", pa.Table.from_pylist([], schema=feature_schema), create=True)
    coverage = epoch_us(con.execute("SELECT min(ts) FROM tx").fetchone()[0])
    # Alert snapshots sort BEFORE events at the same timestamp, ensuring ts < reference_ts.
    timeline = """
        WITH accounts AS (SELECT DISTINCT account_key FROM meta WHERE eligible),
        stream AS (
          SELECT t.from_account_key AS account, t.ts AS moment, 1 AS kind,
            t.source_row_id, NULL::VARCHAR AS uid, 0 AS direction,
            t.to_account_key AS peer, t.amount_paid AS amount,
            t.currency_paid_raw AS currency
          FROM tx t JOIN accounts a ON a.account_key=t.from_account_key
          WHERE NOT t.is_self_transfer AND t.from_account_key<>t.to_account_key
          UNION ALL
          SELECT t.to_account_key,t.ts,1,t.source_row_id,NULL,1,
            t.from_account_key,t.amount_received,t.currency_received_raw
          FROM tx t JOIN accounts a ON a.account_key=t.to_account_key
          WHERE NOT t.is_self_transfer AND t.from_account_key<>t.to_account_key
          UNION ALL
          SELECT account_key,reference_ts,0,-1,training_alert_uid,0,NULL,NULL,NULL
          FROM meta WHERE eligible)
        SELECT * FROM stream ORDER BY account,moment,kind,source_row_id,uid
    """
    account = None
    window = None
    last_ts = None
    buffer = []
    events = snapshots = 0
    for batch in query_batches(con, timeline, 8192):
        for row in batch.to_pylist():
            if row["account"] != account:
                account, window, last_ts = row["account"], Window(86400), None
            moment = epoch_us(row["moment"])
            window.expire(moment)
            if row["kind"]:
                amount = row["amount"] if source.kind == "ibm" else None
                unit = currency_bucket(source.id, row["currency"]) if amount is not None else None
                window.add(Event(moment, row["direction"], row["peer"], amount, unit))
                last_ts = moment
                events += 1
            else:
                if last_ts is not None and last_ts >= moment:
                    raise ValueError("Current/future transaction entered feature history")
                buffer.append(
                    dict(
                        training_alert_uid=row["uid"],
                        **_features(window, moment, last_ts, coverage),
                    )
                )
                snapshots += 1
                if len(buffer) == BATCH:
                    append(con, "features", pa.Table.from_pylist(buffer, schema=feature_schema))
                    buffer.clear()
        if events and events % 100000 < 8192:
            log(folder, "features", history_events=events, snapshots=snapshots)
    if buffer:
        append(con, "features", pa.Table.from_pylist(buffer, schema=feature_schema))
    expected = con.execute("SELECT count(*) FROM meta WHERE eligible").fetchone()[0]
    got = con.execute("SELECT count(*),count(DISTINCT training_alert_uid) FROM features").fetchone()
    if got != (expected, expected):
        raise ValueError("Feature/eligible alert alignment mismatch")
    log(folder, "features_completed", history_events=events, snapshots=snapshots)


def export_dataset(con, source, folder, quality):
    order = " ORDER BY m.reference_ts,m.training_alert_uid"
    fields = ",".join("f." + name for name in FEATURE_NAMES)
    joined = " FROM meta m JOIN features f USING(training_alert_uid) WHERE m.eligible"
    export_query(con, f"SELECT {fields}" + joined + order, folder / "raw.parquet", RAW_SCHEMA)
    export_query(
        con,
        """SELECT t.* FROM input_transactions t
        JOIN (SELECT DISTINCT source_row_id FROM evidence_links) e USING(source_row_id)
        ORDER BY t.source_row_id""",
        folder / "evidence.parquet",
    )
    splits = {
        s: [
            r[0]
            for r in con.execute(
                "SELECT training_alert_uid FROM meta WHERE eligible AND split=? "
                "ORDER BY reference_ts,training_alert_uid",
                [s],
            ).fetchall()
        ]
        for s in ("train", "validation", "test")
    }
    write_json(folder / "splits.json", splits)
    write_json(folder / "quality.json", quality)
    if any(not values for values in splits.values()):
        return dict(
            status="NOT_READY",
            reason="EMPTY_ELIGIBLE_SPLIT",
            rows={s: len(v) for s, v in splits.items()},
        )
    processor = AlertPreprocessor()
    for batch in query_batches(
        con, f"SELECT {fields}" + joined + " AND m.split='train'" + order, 256
    ):
        processor.partial_fit(pa.Table.from_batches([batch]).cast(RAW_SCHEMA), split="train")
    try:
        processor.freeze()
    except ValueError as exc:
        return dict(
            status="NOT_READY", reason=str(exc), rows={s: len(v) for s, v in splits.items()}
        )
    processor.save(folder / "preprocessing.json")
    for split in splits:
        query = f"SELECT {fields}" + joined + f" AND m.split='{split}'" + order
        with ThreeWriter(
            folder / f"X_{split}.parquet",
            pa.schema([(name, pa.float64()) for name in processor.state["feature_columns"]]),
        ) as writer:
            for batch in query_batches(con, query):
                writer.write_table(
                    processor.transform(pa.Table.from_batches([batch]).cast(RAW_SCHEMA))
                )
        export_query(
            con,
            "SELECT m.training_alert_uid,m.y_alert_proxy FROM meta m "
            f"WHERE m.eligible AND m.split='{split}'" + order,
            folder / f"y_{split}.parquet",
            pa.schema([("training_alert_uid", pa.string()), ("y_alert_proxy", pa.int8())]),
        )
        if pq.read_metadata(folder / f"X_{split}.parquet").num_rows != len(splits[split]):
            raise ValueError("X/y row alignment mismatch")
    formats = finish_formats(folder)
    facts = verify_table(folder, "rule_facts")
    return dict(
        status="COMPLETED",
        rows={s: len(v) for s, v in splits.items()},
        feature_columns=processor.state["feature_columns"],
        format_tables=formats["tables"],
        rule_facts_verification=facts,
    )
