#!/usr/bin/env python3
"""Zero-Shot LLM Semantic Prior Generator (ICLR 2027 pivot).

Prompts a large language model to act as the domain expert that our
Neurosynth meta-analysis maps were before: given the 116 AAL region labels
and a cognitive task name, the model assigns each region a continuous
relevance score in [0, 1] based on neuroscience literature.  The parsed
scores are min-max normalized and written in *exactly* the schema of the
existing Neurosynth priors (scripts/02_build_prior_maps.py):

    outputs/priors/llm/{task_slug}/roi_prior.csv
        roi_index, roi_label, raw_score, prior_score

so every downstream consumer (load_roi_prior in scripts/40-42, the
faithfulness protocol, control-prior builders) works unchanged.

Providers
---------
- ollama (default): local HTTP API at --ollama-url (no key required);
  default model ``llama3``.
- openai: uses the ``openai`` python package when installed, otherwise the
  REST API via ``requests``; default model ``gpt-4o-mini``; requires
  OPENAI_API_KEY in the environment.

Robustness
----------
- Strict JSON instruction + ``format="json"`` (Ollama) / JSON response
  format (OpenAI); a repair pass strips code fences and extracts the first
  JSON object if the model adds prose.
- Case-insensitive label matching against the atlas; missing labels raise
  unless ``--fill-missing 0.0`` is passed explicitly (then they are filled
  and reported as warnings - never silently).
- Scores are clamped to [0, 1].
- ``--dry-run`` prints the prompt and exits without any network call.

Controls (for the true/shuffled/random safeguard)
-------------------------------------------------
``--controls`` additionally writes an anatomically shuffled variant
(outputs/priors/llm/{task_slug}_shuffled/) by permuting the LLM scores
across ROIs with a fixed seed; the random control remains the canonical
outputs/priors/random_prior/aal116/roi_prior.csv shared by all methods.

Example
-------
    python scripts/46_generate_llm_priors.py --task "Working Memory" \
        --provider ollama --model llama3 --controls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

DEFAULT_LABELS = "inputs/atlases/AAL116_labels.csv"
DEFAULT_OUT_ROOT = "outputs/priors/llm"
CANONICAL_RANDOM_PRIOR = "outputs/priors/random_prior/aal116/roi_prior.csv"

PROMPT_TEMPLATE = """You are an expert cognitive neuroscientist with deep knowledge of \
human functional neuroimaging literature (fMRI/PET activation studies, lesion data, \
meta-analyses such as Neurosynth and BrainMap).

Task: rate how strongly each of the following {n_rois} brain regions (AAL atlas labels) \
is implicated in the cognitive domain "{task}".

Instructions:
- Assign EACH region exactly one continuous relevance score between 0.0 and 1.0.
- 1.0 = core hub repeatedly reported in imaging studies of this domain;
- around 0.5 = moderate / supporting involvement;
- 0.0 = no meaningful association with the domain.
- Consider the hemisphere suffixes (_L / _R) and use lateralization knowledge where \
the literature supports it.
- Base ratings on established neuroscience literature, not on the labels alone.
- Respond with STRICT JSON only - no prose before or after - using exactly this schema:
{{"scores": {{"<AAL label>": <float>, ...}}}} covering ALL {n_rois} regions listed below.

Regions:
{region_list}
"""


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or "task"


def load_region_labels(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"roi_index", "roi_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    if df.roi_label.duplicated().any():
        dupes = sorted(df.roi_label[df.roi_label.duplicated()].tolist())
        raise ValueError(f"{path} has duplicate labels: {dupes}")
    return df.sort_values("roi_index").reset_index(drop=True)


def build_prompt(task: str, labels: List[str]) -> str:
    region_list = "\n".join(f"- {label}" for label in labels)
    return PROMPT_TEMPLATE.format(n_rois=len(labels), task=task, region_list=region_list)


# ---------------------------------------------------------------------------
# Provider backends - each returns the raw assistant text
# ---------------------------------------------------------------------------
def _chat_ollama(prompt: str, model: str, base_url: str, temperature: float,
                 seed: int, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "seed": seed},
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat", json=payload, timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["message"]["content"])


def _chat_openai_sdk(prompt: str, model: str, temperature: float, seed: int,
                     timeout: float) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The 'openai' package is not installed; pip install openai "
            "(or use --provider ollama)."
        ) from exc
    client = OpenAI(timeout=timeout)
    kwargs: Dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You respond with strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if seed > 0:
        kwargs["seed"] = seed
    return str(client.chat.completions.create(**kwargs).choices[0].message.content)


def _chat_openai_rest(prompt: str, model: str, temperature: float, seed: int,
                      timeout: float) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
    body: Dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You respond with strict JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    if seed > 0:
        body["seed"] = seed
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body, timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json()["choices"][0]["message"]["content"])


def call_llm(provider: str, prompt: str, model: str, ollama_url: str,
             temperature: float, seed: int, timeout: float) -> str:
    if provider == "ollama":
        return _chat_ollama(prompt, model, ollama_url, temperature, seed, timeout)
    try:
        return _chat_openai_sdk(prompt, model, temperature, seed, timeout)
    except ImportError:
        return _chat_openai_rest(prompt, model, temperature, seed, timeout)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_object(text: str) -> Dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match is None:
        raise ValueError(f"No JSON object found in model response:\n{text[:500]}")
    return json.loads(match.group(0))


def parse_scores(payload: Dict, labels: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Map the model's JSON onto the atlas order (case-insensitive keys)."""
    raw = payload.get("scores", payload)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a 'scores' object, got {type(raw).__name__}")
    lookup = {str(k).strip().lower(): v for k, v in raw.items()}
    scores = np.full(len(labels), np.nan)
    unmatched_model_keys = []
    matched = set()
    for i, label in enumerate(labels):
        value = lookup.get(label.lower())
        if value is None:
            continue
        try:
            scores[i] = float(value)
        except (TypeError, ValueError):
            continue
        matched.add(label.lower())
    unmatched_model_keys = [k for k in lookup if k not in matched]
    missing = [labels[i] for i in range(len(labels)) if not np.isfinite(scores[i])]
    return scores, missing


# ---------------------------------------------------------------------------
# Output writers (schema identical to scripts/02_build_prior_maps.py)
# ---------------------------------------------------------------------------
def write_prior_csv(df: pd.DataFrame, scores: np.ndarray, path: Path) -> None:
    out = pd.DataFrame({
        "roi_index": df.roi_index.astype(int),
        "roi_label": df.roi_label.astype(str),
        "raw_score": scores.astype(np.float64),
    })
    low, high = float(out.raw_score.min()), float(out.raw_score.max())
    if high - low < 1e-12:
        raise ValueError(
            "LLM scores are constant; refusing to write a degenerate prior."
        )
    out["prior_score"] = (out.raw_score - low) / (high - low)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, help='Cognitive task, e.g. "Working Memory"')
    ap.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--model", default=None,
                    help="llama3 (ollama default) or gpt-4o-mini (openai default)")
    ap.add_argument("--labels", default=DEFAULT_LABELS)
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--slug", default=None, help="Output directory name override")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--fill-missing", type=float, default=None,
                    help="Fill unreturned regions with this score instead of failing")
    ap.add_argument("--controls", action="store_true",
                    help="Also write an anatomically shuffled control prior")
    ap.add_argument("--dry-run", action="store_true", help="Print prompt and exit")
    args = ap.parse_args()

    model = args.model or ("llama3" if args.provider == "ollama" else "gpt-4o-mini")
    slug = args.slug or slugify(args.task)

    df = load_region_labels(args.labels)
    labels = df.roi_label.tolist()
    prompt = build_prompt(args.task, labels)

    if args.dry_run:
        print(prompt)
        print(f"[dry-run] provider={args.provider} model={model} "
              f"out={Path(args.out_root) / slug / 'roi_prior.csv'}", file=sys.stderr)
        return

    print(f"Generating '{args.task}' prior with {args.provider}:{model} "
          f"({len(labels)} regions)...", flush=True)
    last_error: Exception | None = None
    for attempt in range(1, max(args.retries, 1) + 1):
        try:
            text = call_llm(
                args.provider, prompt, model, args.ollama_url,
                args.temperature, args.seed, args.timeout,
            )
            payload = extract_json_object(text)
            break
        except Exception as exc:  # noqa: BLE001 - retry any backend failure
            last_error = exc
            print(f"[attempt {attempt}/{args.retries}] failed: {exc}", flush=True)
            time.sleep(2.0 * attempt)
    else:
        raise RuntimeError(f"All {args.retries} attempts failed") from last_error

    scores, missing = parse_scores(payload, labels)
    if missing:
        if args.fill_missing is None:
            raise ValueError(
                f"{len(missing)} regions missing from the LLM response "
                f"(e.g. {missing[:5]}...). Pass --fill-missing 0.0 to fill them."
            )
        scores[~np.isfinite(scores)] = float(args.fill_missing)
        print(f"[WARN] filled {len(missing)} missing regions with "
              f"{args.fill_missing}: {missing}", file=sys.stderr)
    scores = np.clip(scores, 0.0, 1.0)
    print(f"Parsed {int(np.isfinite(scores).sum())}/{len(labels)} region scores; "
          f"range [{scores.min():.3f}, {scores.max():.3f}]")

    out_dir = Path(args.out_root) / slug
    write_prior_csv(df, scores, out_dir / "roi_prior.csv")

    provenance = {
        "task": args.task,
        "slug": slug,
        "provider": args.provider,
        "model": model,
        "temperature": args.temperature,
        "seed": args.seed,
        "n_regions": len(labels),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fill_missing": args.fill_missing,
        "missing_regions": missing,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out_dir / 'roi_prior.csv'}")

    if args.controls:
        rng = np.random.default_rng(args.seed)
        shuffled = scores[rng.permutation(len(scores))]
        shuf_dir = Path(args.out_root) / f"{slug}_shuffled"
        write_prior_csv(df, shuffled, shuf_dir / "roi_prior.csv")
        (shuf_dir / "provenance.json").write_text(json.dumps({
            **provenance,
            "control_of": slug,
            "control_type": "anatomically_shuffled",
            "shuffle_seed": args.seed,
        }, indent=2), encoding="utf-8")
        print(f"Wrote {shuf_dir / 'roi_prior.csv'} (shuffled control); "
              f"use the canonical {CANONICAL_RANDOM_PRIOR} as the random control.")


if __name__ == "__main__":
    main()
