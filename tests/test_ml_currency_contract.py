from aml_ml.schema import FEATURE_VERSION, currency_bucket


def test_updated_currency_feature_contract_preserves_native_unknowns():
    expected = {"Yuan": "CNY", "Shekel": "ILS", "Ruble": "RUB", "Rupee": "INR"}
    for raw, code in expected.items():
        assert currency_bucket("ibm_hi_small", raw) == code
    assert currency_bucket("ibm_hi_small", "Bitcoin") == "CRYPTO:BTC"
    assert currency_bucket("saml_d", "Dirham") != currency_bucket("ibm_hi_small", "Dirham")
    assert currency_bucket("saml_d", "UNKNOWN") is None
    assert "v2-currency-20261030" in FEATURE_VERSION
