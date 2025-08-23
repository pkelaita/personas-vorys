import os
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import AsyncAzureOpenAI


def is_azure_model(model: str) -> Tuple[bool, str]:
    """
    Returns (is_azure, deployment_or_model).
    If model starts with 'azure:', the remainder is treated as the Azure deployment name.
    """
    if not isinstance(model, str):
        return False, str(model)
    if model.lower().startswith("azure:"):
        return True, model.split(":", 1)[1].strip()
    return False, model


def normalize_model_label(model: str) -> str:
    """
    Normalizes a model string for safe filenames.
    Example: 'azure:gpt-4o' -> 'azure-gpt-4o'
    """
    return str(model).replace(":", "-")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


async def azure_chat(
    *,
    system_prompt: str,
    prompt: str,
    deployment: str,
    timeout: Optional[int] = None,
    temperature: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
) -> str:
    """
    Calls an Azure OpenAI Chat Completions deployment asynchronously and returns the text content.
    Requires:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT (e.g. https://<resource-name>.openai.azure.com)
      - AZURE_OPENAI_API_VERSION (e.g. 2024-06-01)
    The 'deployment' parameter should match your Azure deployment name for the chosen model.
    """
    api_key = _require_env("AZURE_OPENAI_API_KEY")
    endpoint = _require_env("AZURE_OPENAI_ENDPOINT")
    api_version = _require_env("AZURE_OPENAI_API_VERSION")

    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )

    params: Dict[str, Any] = {
        "model": deployment,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "timeout": timeout,
    }
    if temperature is not None:
        params["temperature"] = temperature

    if reasoning_effort is not None:
        eff = str(reasoning_effort).strip().lower()
        if eff in {"low", "medium", "high", "minimal"}:
            extra_body = params.get("extra_body") or {}
            # Azure Chat Completions expects 'reasoning_effort' (string) in request body for reasoning models
            extra_body["reasoning_effort"] = eff
            params["extra_body"] = extra_body

    resp = await client.chat.completions.create(**params)

    content: Union[str, List[Dict[str, Any]], None] = None
    if resp.choices and resp.choices[0].message:
        content = resp.choices[0].message.content

    if content is None:
        return ""

    # If the SDK returns a list of content parts, join any text fields
    if isinstance(content, list):
        parts = []
        for part in content:
            # part may be a dict with a 'text' field or similar
            if isinstance(part, dict):
                txt = part.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts).strip()

    if isinstance(content, str):
        return content.strip()

    # Fallback to string representation
    return str(content).strip()
