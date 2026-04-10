"""
UZL OpenAI-compatible model for ReSLLM.

Handles reasoning models (like gpt-oss-120b) where the logprobs stream
contains reasoning tokens before the actual answer. The content answer
appears after the <|channel|>final<|message|> marker sequence.
"""

import hashlib
import json
import math
import os
import time
from pathlib import Path
from openai import OpenAI, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError
from .base_model import BaseModel

MAX_RETRIES = 5
RETRY_DELAYS = [2, 5, 10, 20, 40]

_DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "cache"
_CACHE_FILE = "predictions.json"


class UZLModel(BaseModel):
    """OpenAI-compatible model pointed at the UZL endpoint."""

    def load_model(self, base_url=None, api_key=None, model_name=None, cache_dir=None):
        self.base_url = base_url or os.environ.get("UZL_BASE_URL", "")
        self.api_key_str = api_key or os.environ.get("UZL_API_KEY", self.api_key or "")
        self.model_name = model_name or os.environ.get("UZL_MODEL", "gpt-oss-120b")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key_str)

        # Load prediction cache
        self._cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_path = self._cache_dir / _CACHE_FILE
        if self._cache_path.exists():
            with open(self._cache_path) as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    def _cache_key(self, prompt_text):
        return hashlib.sha256(prompt_text.encode()).hexdigest()

    def _save_cache(self):
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def predict(self, query, source_prompt):
        """
        Score a single (query, resource) pair.

        Returns score(q, r) = logprob(yes) - logprob(no) following the
        ReSLLM paper (Equation 1). Falls back to logprob(yes) alone if
        "no" is not in the top logprobs.
        """
        prompt_text = source_prompt.format(query=query)
        key = self._cache_key(prompt_text)

        if key in self._cache:
            return self._cache[key]["score"]

        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=200,
                    temperature=0,
                    logprobs=True,
                    top_logprobs=5,
                )
                break
            except (APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                delay = RETRY_DELAYS[attempt]
                print(f"  API error ({e.__class__.__name__}), retrying in {delay}s...")
                time.sleep(delay)

        choice = response.choices[0]
        content = (choice.message.content or "").strip().lower()
        logprobs_data = choice.logprobs

        # Try to extract logprobs from the answer token
        score = None
        if logprobs_data and logprobs_data.content:
            yes_logprob, no_logprob = self._extract_answer_logprobs(
                logprobs_data.content
            )
            if yes_logprob is not None:
                if no_logprob is not None:
                    score = yes_logprob - no_logprob
                else:
                    score = yes_logprob

        # Fallback: parse text content
        if score is None:
            if content.startswith("yes"):
                score = 0.0
            elif content.startswith("no"):
                score = -1.0
            else:
                score = -2.0

        self._cache[key] = {"prompt_preview": prompt_text[:200], "score": score}
        self._save_cache()
        return score

    def _extract_answer_logprobs(self, content_logprobs):
        """
        Find the actual answer token in the logprobs stream.

        For reasoning models, the stream looks like:
          <|channel|> analysis <|message|> [reasoning...] <|end|>
          <|start|> assistant <|channel|> final <|message|> yes <|return|>

        We look for the token after 'final' + '<|message|>' to find the answer.
        """
        yes_logprob = None
        no_logprob = None

        # Strategy 1: Find the answer after <|channel|>final<|message|> sequence
        for i in range(len(content_logprobs) - 1):
            token = content_logprobs[i].token
            if token == "final" and i + 2 < len(content_logprobs):
                next_token = content_logprobs[i + 1].token
                if next_token == "<|message|>":
                    answer_entry = content_logprobs[i + 2]
                    yes_logprob, no_logprob = self._find_yes_no_in_top_logprobs(
                        answer_entry
                    )
                    if yes_logprob is not None:
                        return yes_logprob, no_logprob

        # Strategy 2: Check the last few non-special tokens
        for entry in reversed(content_logprobs):
            if entry.token.startswith("<|") and entry.token.endswith("|>"):
                continue
            yes_logprob, no_logprob = self._find_yes_no_in_top_logprobs(entry)
            if yes_logprob is not None:
                return yes_logprob, no_logprob
            break  # only check the last real token

        return None, None

    def _find_yes_no_in_top_logprobs(self, logprob_entry):
        """Extract yes and no logprobs from a token's top_logprobs."""
        yes_logprob = None
        no_logprob = None

        if not logprob_entry.top_logprobs:
            # Use the token itself
            token_lower = logprob_entry.token.strip().lower()
            if token_lower == "yes":
                return logprob_entry.logprob, None
            return None, None

        for tp in logprob_entry.top_logprobs:
            token_lower = tp.token.strip().lower()
            if token_lower == "yes" and yes_logprob is None:
                yes_logprob = tp.logprob
            elif token_lower == "no" and no_logprob is None:
                no_logprob = tp.logprob

        return yes_logprob, no_logprob
