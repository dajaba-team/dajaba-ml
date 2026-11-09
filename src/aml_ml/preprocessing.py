"""train 전용 결측 대체·어휘·상수 열 선택을 수행하고 재사용하도록 직렬화합니다."""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as _compute

from .schema import CATEGORICAL, FEATURE_VERSION, FORBIDDEN, model_numeric

pc = cast(Any, _compute)


class Preprocessor:
    def __init__(
        self,
        *,
        standardize=False,
        max_categories=128,
        primary_windows=("24h",),
        numeric=None,
        categorical=None,
        feature_version=FEATURE_VERSION,
    ):
        self.numeric = model_numeric(primary_windows) if numeric is None else tuple(numeric)
        self.categorical = CATEGORICAL if categorical is None else tuple(categorical)
        self.feature_version = feature_version
        if len(set(self.numeric + self.categorical)) != len(self.numeric + self.categorical):
            raise ValueError("Duplicate feature names")
        self.standardize = standardize
        self.max_categories = max_categories
        self.stats = {name: dict(n=0, mean=0.0, m2=0.0, missing=0, rows=0) for name in self.numeric}
        self.categories = {name: Counter() for name in self.categorical}
        self.state: dict[str, Any] | None = None

    def partial_fit(self, table: pa.Table, *, split: str):
        if split != "train":
            raise ValueError("Preprocessor can fit on train only")
        if self.state is not None:
            raise ValueError("Frozen preprocessing state cannot be refitted")
        if set(self.numeric + self.categorical) & FORBIDDEN:
            raise ValueError("Feature allowlist contains a forbidden label/identity column")
        for name in self.numeric:
            values = pc.cast(table.column(name), pa.float64())
            n = len(values) - values.null_count
            stats = self.stats[name]
            stats["rows"] += len(values)
            stats["missing"] += values.null_count
            if not n:
                continue
            if not pc.all(pc.fill_null(pc.is_finite(values), True)).as_py():
                raise ValueError(f"Nonfinite training feature: {name}")
            mean = pc.mean(values).as_py()
            m2 = pc.variance(values, ddof=0).as_py() * n
            old = stats["n"]
            combined = old + n
            delta = mean - stats["mean"]
            stats["mean"] += delta * n / combined
            stats["m2"] += m2 + delta * delta * old * n / combined
            stats["n"] = combined
        for name in self.categorical:
            self.categories[name].update(
                {
                    row["values"]: row["counts"]
                    for row in pc.value_counts(table.column(name)).to_pylist()
                }
            )

    def freeze(self) -> dict:
        if not self.stats or not next(iter(self.stats.values()))["rows"]:
            raise ValueError("No training rows to fit preprocessing")
        numeric = {}
        categories = {}
        columns = []
        for name, item in self.stats.items():
            variance = max(0.0, item["m2"] / item["n"]) if item["n"] else 0.0
            active = item["n"] > 0 and variance > 1e-18
            missing_active = 0 < item["missing"] < item["rows"]
            numeric[name] = dict(
                mean=item["mean"] if item["n"] else 0.0,
                std=math.sqrt(variance) or 1.0,
                observed=item["n"],
                missing=item["missing"],
                active=active,
                missing_indicator_active=missing_active,
                fallback="NO_OBSERVED_TRAIN_VALUE" if not item["n"] else None,
            )
            if active:
                columns.append(name)
            if missing_active:
                columns.append(name + "__missing")
        for name, counts in self.categories.items():
            known = [(key, value) for key, value in counts.items() if key is not None]
            known.sort(key=lambda pair: (-pair[1], pair[0]))
            mapping = {key: i + 2 for i, (key, _) in enumerate(known[: self.max_categories])}
            active = len(counts) > 1
            categories[name] = dict(
                mapping=mapping,
                unknown_code=0,
                missing_code=1,
                active=active,
                train_distinct=len(counts),
            )
            if active:
                columns.append(name + "__category")
        if not columns:
            raise ValueError("No variable train feature remains")
        self.state = dict(
            feature_version=self.feature_version,
            fit_scope="TRAIN_ONLY",
            fit_rows=next(iter(self.stats.values()))["rows"],
            standardize=self.standardize,
            numeric=numeric,
            categorical=categories,
            feature_columns=columns,
            dropped_constant_features=[name for name, v in numeric.items() if not v["active"]],
            note="All raw features remain in the raw dataset; constant selection and vocabulary see train only",
        )
        return self.state

    def transform(self, table: pa.Table) -> pa.Table:
        if self.state is None:
            raise ValueError("Preprocessing is not fitted")
        arrays = {}
        for name, spec in self.state["numeric"].items():
            col = pc.cast(table.column(name), pa.float64())
            if not pc.all(pc.fill_null(pc.is_finite(col), True)).as_py():
                raise ValueError(f"Nonfinite feature: {name}")
            if spec["active"]:
                values = pc.fill_null(col, spec["mean"])
                if self.state["standardize"]:
                    values = pc.divide(pc.subtract(values, spec["mean"]), spec["std"])
                arrays[name] = values
            if spec["missing_indicator_active"]:
                arrays[name + "__missing"] = pc.cast(pc.is_null(col), pa.float64())
        for name, spec in self.state["categorical"].items():
            if spec["active"]:
                arrays[name + "__category"] = pa.array(
                    [
                        1.0 if value is None else float(spec["mapping"].get(value, 0))
                        for value in table.column(name).to_pylist()
                    ],
                    type=pa.float64(),
                )
        result = pa.Table.from_arrays(
            [arrays[name] for name in self.state["feature_columns"]],
            names=self.state["feature_columns"],
        )
        return result

    def save(self, path: Path):
        if self.state is None:
            raise ValueError("No fitted state")
        path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        state = json.loads(path.read_text(encoding="utf-8"))
        if state["feature_version"] != FEATURE_VERSION or state["fit_scope"] != "TRAIN_ONLY":
            raise ValueError("Unsupported preprocessing artifact")
        result = cls()
        result.state = state
        return result
