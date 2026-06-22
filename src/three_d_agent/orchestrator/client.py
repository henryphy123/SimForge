import os
import anthropic

DEFAULT_BASE_URL = "https://cp.compshare.cn"
DEFAULT_MODEL = "glm-5.2"


def resolve_base_url() -> str:
    return (
        os.environ.get("THREE_D_AGENT_BASE_URL")
        or os.environ.get("ANTHROPIC_BASE_URL")
        or DEFAULT_BASE_URL
    )


def resolve_api_key() -> str | None:
    return os.environ.get("THREE_D_AGENT_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


def build_client(
    base_url: str | None = None, api_key: str | None = None
) -> anthropic.Anthropic:
    """Construct an Anthropic-compatible client for the configured provider.

    The key is read from the environment (never hardcoded). Raises if absent so
    a missing credential fails loudly instead of hitting the API unauthenticated.
    """
    key = api_key or resolve_api_key()
    if not key:
        raise RuntimeError(
            "No API key found. Set THREE_D_AGENT_API_KEY (or ANTHROPIC_API_KEY) "
            "in your environment before running."
        )
    return anthropic.Anthropic(base_url=base_url or resolve_base_url(), api_key=key)
