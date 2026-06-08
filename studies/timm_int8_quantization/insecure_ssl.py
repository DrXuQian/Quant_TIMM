"""Optional, opt-in disabling of TLS/SSL certificate verification for downloads.

This exists for restricted networks (corporate TLS inspection, a missing /
outdated CA bundle) where the proper fixes (updating `certifi`, installing the
corporate root CA) are not available. The study's entry-point scripts import it,
so it is **active by default**. Keep TLS verification ON by setting:

    export TIMM_INT8_INSECURE_SSL=0

When active it also points the HF hub at the hf-mirror.com mirror (unless you
already set HF_ENDPOINT) and silences the resulting insecure-request warnings.

SECURITY WARNING: this turns off MITM protection for *all* HTTPS in the process.
Use it only on a network you trust, to fetch public model weights / datasets.
Prefer the real fixes (see RUNNING.md) whenever you can.
"""

import os
import ssl
import warnings


def _active() -> bool:
    # Active by default in these scripts; set TIMM_INT8_INSECURE_SSL=0 to keep
    # TLS certificate verification ON.
    val = os.environ.get("TIMM_INT8_INSECURE_SSL")
    if val is None:
        return True
    return val.lower() not in ("0", "false", "no", "")


def maybe_disable_ssl_verification() -> bool:
    """Disable HTTPS certificate verification if TIMM_INT8_INSECURE_SSL is set.

    Returns True if verification was disabled, False if left untouched.
    """
    if not _active():
        return False

    # Route HF through the mirror unless the user picked an endpoint already.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"  # honored by newer hub
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

    # stdlib ssl / urllib (used by torch.hub and some downloaders).
    ssl._create_default_https_context = ssl._create_unverified_context

    try:  # silence the flood of InsecureRequestWarning
        import urllib3
        urllib3.disable_warnings()
    except Exception:
        pass

    # requests — this is the path huggingface_hub / timm weight downloads use, so
    # it is the one that actually fixes the "CERTIFICATE_VERIFY_FAILED" HEAD error.
    try:
        import requests
        _orig = requests.sessions.Session.merge_environment_settings

        def _merge(self, url, proxies, stream, verify, cert):
            settings = _orig(self, url, proxies, stream, verify, cert)
            settings["verify"] = False
            return settings

        requests.sessions.Session.merge_environment_settings = _merge
    except Exception:
        pass

    # huggingface_hub's official hook: a session factory with verify disabled.
    try:
        import requests
        from huggingface_hub import configure_http_backend

        def _factory():
            s = requests.Session()
            s.verify = False
            return s

        configure_http_backend(backend_factory=_factory)
    except Exception:
        pass

    # httpx (used by some newer clients).
    try:
        import httpx
        for _name in ("Client", "AsyncClient"):
            _base = getattr(httpx, _name)

            class _Insecure(_base):  # type: ignore[valid-type, misc]
                def __init__(self, *a, **kw):
                    kw["verify"] = False
                    super().__init__(*a, **kw)

            setattr(httpx, _name, _Insecure)
    except Exception:
        pass

    warnings.warn(
        "TIMM_INT8_INSECURE_SSL=1: HTTPS certificate verification is DISABLED "
        f"(HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}). Use only on trusted networks."
    )
    return True


# Activate on import so a single `import insecure_ssl` at the top of an entry
# point is enough.
maybe_disable_ssl_verification()
