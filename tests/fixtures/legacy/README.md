# Historical test fixtures

These inputs test historical behavior, not current production policy. N=10, offline version identifiers, pending nulls, and fail-closed semantics are preserved. Current R01/R02 use N=5.

Private evidence links and the live input path were replaced with explicit test markers. Corresponding reference-string assertions use the same marker; numerical and failure assertions were not weakened. Original operational configs and completed-run provenance remain unchanged.