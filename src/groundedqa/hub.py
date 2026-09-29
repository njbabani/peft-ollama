"""Pin Hugging Face inputs to immutable commit SHAs for reproducible runs."""

import re
import time


def resolve_hub_revision(
    lookup,
    repo_id: str,
    revision: str,
    retry_errors: tuple,
    *,
    attempts: int = 5,
    sleep=time.sleep,
) -> str:
    """Resolve an immutable commit; retry only transport and transient HTTP errors."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        return revision.lower()
    for attempt in range(attempts):
        try:
            info = lookup(repo_id, revision=revision, timeout=30)
        except retry_errors as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in {408, 429, 500, 502, 503, 504}:
                raise
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Hugging Face lookup failed for {repo_id}@{revision} after "
                    f"{attempts} attempts. Retry later or pin a 40-character commit SHA."
                ) from exc
            delay = min(2 ** (attempt + 1), 30)
            print(f"Hub lookup {repo_id}: transient {status or 'transport error'}; retrying in {delay}s")
            sleep(delay)
            continue
        sha = getattr(info, "sha", None)
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            raise ValueError(f"Hub returned no valid commit SHA for {repo_id}@{revision}")
        return sha.lower()
    raise AssertionError("Unreachable retry state")


def resolve_all(config) -> dict[str, str]:
    """Commit SHAs for the model, dataset, embedder and reranker in a Config."""
    import httpx
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    errors = (HfHubHTTPError, httpx.TransportError)
    return {
        "model": resolve_hub_revision(
            api.model_info, config.model.model_id, config.model.model_revision, errors
        ),
        "dataset": resolve_hub_revision(
            api.dataset_info, config.data.dataset_id, config.data.dataset_revision, errors
        ),
        "embedding": resolve_hub_revision(
            api.model_info,
            config.retrieval.embedding_id,
            config.retrieval.embedding_revision,
            errors,
        ),
        "reranker": resolve_hub_revision(
            api.model_info,
            config.retrieval.reranker_id,
            config.retrieval.reranker_revision,
            errors,
        ),
    }
