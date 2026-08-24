import hashlib
from typing import Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

_SUPPORTED_MODEL_PROXY_SCHEMES = frozenset({"http", "https", "socks5", "socks5h"})


def validate_model_proxy(proxy: str) -> str:
    if proxy.startswith("os.environ/"):
        return proxy
    if not proxy.strip():
        raise ValueError("Model proxy must be a non-empty URL")
    try:
        parsed = urlsplit(proxy)
        hostname = parsed.hostname
        parsed.port
    except ValueError as exc:
        raise ValueError("Model proxy URL is invalid") from exc
    if parsed.scheme.lower() not in _SUPPORTED_MODEL_PROXY_SCHEMES or hostname is None:
        raise ValueError("Model proxy must use http, https, socks5, or socks5h and include a host")
    if parsed.query or parsed.fragment:
        raise ValueError("Model proxy URL cannot include a query or fragment")
    return proxy


def resolve_model_proxy(proxy: str) -> str:
    if proxy.startswith("os.environ/"):
        from litellm.secret_managers.main import get_secret_str

        resolved_proxy = get_secret_str(proxy)
        if resolved_proxy is None or not resolved_proxy.strip():
            raise ValueError("Model proxy environment variable is not set")
        proxy = resolved_proxy
    return validate_model_proxy(proxy)


def model_proxy_fingerprint(proxy: Optional[str]) -> Optional[str]:
    if proxy is None:
        return None
    return hashlib.sha256(proxy.encode()).hexdigest()


def mask_model_proxy(proxy: str) -> str:
    try:
        parsed = urlsplit(proxy)
        if parsed.username is None:
            return proxy
        hostname = parsed.hostname
        if hostname is None:
            return "***"
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        credentials = "***:***" if parsed.password is not None else "***"
        return urlunsplit(
            SplitResult(
                scheme=parsed.scheme,
                netloc=f"{credentials}@{host}",
                path=parsed.path,
                query=parsed.query,
                fragment=parsed.fragment,
            )
        )
    except ValueError:
        return "***"
