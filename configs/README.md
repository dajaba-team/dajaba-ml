# Public configuration examples

Only *.example.json is published. Existing operational configs stay local and retain their provenance.

offline_rules.example.json preserves approved N=5 multi-rule parameters, offline version identifiers and source split boundaries. It is a template, not a new execution record or backend deployment attestation. Stale execution-status descriptions are omitted only from this new example.

Copy assembly.example.json to assembly.local.json and replace every placeholder with the actual verified assembly run directory and actual run.json SHA-256. The placeholder strings are intentionally not valid evidence. Do not bypass hash verification. Then select offline_rules.example.json explicitly for a separately approved run; this documentation does not initiate preprocessing.

Historical N=10 and pending policies are test fixtures under tests/fixtures/legacy, not current defaults. SAML remains excluded from R07.