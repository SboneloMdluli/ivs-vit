"""Smoke tests: package imports without errors."""


def test_package_import() -> None:
    import implied_volatility_diffusion as ivd

    assert ivd.__version__
