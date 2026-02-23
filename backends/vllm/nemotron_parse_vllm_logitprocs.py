"""
Custom vLLM v1 logits processors for NVIDIA Nemotron Parse.

Tuning via environment variables:
  - NEMOTRON_PARSE_TABLE_PREFIX (default: \\begin{tabular})
  - NEMOTRON_PARSE_REP_MAX      (default: 10)
  - NEMOTRON_PARSE_REP_WINDOW   (default: 2 * max_ngram * (rep_max + 1))
  - NEMOTRON_PARSE_REP_NGRAMS   (default: 1,2,3,4,5)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Set

import torch
from transformers import AutoTokenizer

from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import BatchUpdate, LogitsProcessor


def _strip_trailing_negative_token_ids(token_ids: List[int]) -> List[int]:
    """vLLM v1 keeps a trailing -1 placeholder in output_token_ids."""
    i = len(token_ids)
    while i > 0 and token_ids[i - 1] < 0:
        i -= 1
    return token_ids[:i]


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_csv_ints(name: str, default: List[int]) -> List[int]:
    val = os.environ.get(name, "").strip()
    if not val:
        return default
    out: List[int] = []
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return default
    return out or default


def _warn_if_unexpected_arch(vllm_config) -> None:
    try:
        arches = getattr(vllm_config.model_config, "architectures", None) or []
    except Exception:
        arches = []
    if any(a == "NemotronParseForConditionalGeneration" for a in arches):
        return
    warnings.warn(
        "Nemotron-Parse logits processors enabled for a model whose "
        f"architectures={arches!r}. These processors assume Nemotron-Parse-style "
        "special tokens like <x_...>, <y_...>, and <class_...>; behavior may be "
        "unsupported for other models.",
        RuntimeWarning,
        stacklevel=2,
    )


def _load_hf_tokenizer(vllm_config):
    model_cfg = vllm_config.model_config
    tok_name = getattr(model_cfg, "tokenizer", None) or getattr(model_cfg, "model", None)
    trust_rc = bool(getattr(model_cfg, "trust_remote_code", False))
    revision = getattr(model_cfg, "tokenizer_revision", None) or getattr(model_cfg, "revision", None)
    token = getattr(model_cfg, "hf_token", None)
    kwargs = {"trust_remote_code": trust_rc}
    if revision:
        kwargs["revision"] = revision
    if token:
        kwargs["token"] = token
    return AutoTokenizer.from_pretrained(tok_name, **kwargs)


def _build_token_sets(tokenizer) -> tuple[Set[int], Set[int], Set[int]]:
    """Build token-id sets for <x_...>, <y_...>, and <class_...>."""

    def _scan(items) -> tuple[Set[int], Set[int], Set[int]]:
        x_ids: Set[int] = set()
        y_ids: Set[int] = set()
        class_ids: Set[int] = set()
        for token, tid in items:
            if token.startswith("<x_") and token.endswith(">") and token.count(">") == 1:
                x_ids.add(tid)
            elif token.startswith("<y_") and token.endswith(">") and token.count(">") == 1:
                y_ids.add(tid)
            elif token.startswith("<class_") and token.endswith(">") and token.count(">") == 1:
                class_ids.add(tid)
        return x_ids, y_ids, class_ids

    added_vocab = {}
    if hasattr(tokenizer, "get_added_vocab"):
        try:
            added_vocab = tokenizer.get_added_vocab() or {}
        except Exception:
            added_vocab = {}

    x_ids, y_ids, class_ids = _scan(added_vocab.items())

    if len(x_ids) < 100 or len(y_ids) < 100 or not class_ids:
        vocab = tokenizer.get_vocab()
        x2, y2, c2 = _scan(vocab.items())
        x_ids, y_ids, class_ids = x2, y2, c2

    return x_ids, y_ids, class_ids


@dataclass
class _TableReqState:
    output_ids: List[int]
    insertion_active: bool = False
    insertion_pos: int = 0
    expecting_end_coords: bool = False
    _last_class_pos: int = -1


class NemotronParseTableInsertionLogitsProcessor(LogitsProcessor):
    """Force a table prefix right after each START <x_...><y_...> coordinate pair."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        self.enabled = True
        self.req_states: Dict[int, _TableReqState] = {}

        self.table_prefix = os.environ.get("NEMOTRON_PARSE_TABLE_PREFIX", r"\begin{tabular}")
        self.table_prefix_ids: List[int] = []
        self.x_token_ids: Set[int] = set()
        self.y_token_ids: Set[int] = set()
        self.class_token_ids: Set[int] = set()

        if self.enabled:
            _warn_if_unexpected_arch(vllm_config)
            tok = _load_hf_tokenizer(vllm_config)
            self.table_prefix_ids = tok.encode(self.table_prefix, add_special_tokens=False)
            self.x_token_ids, self.y_token_ids, self.class_token_ids = _build_token_sets(tok)

        if not self.x_token_ids or not self.y_token_ids or not self.table_prefix_ids:
            self.enabled = False

    @classmethod
    def validate_params(cls, sampling_params):
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not self.enabled:
            return

        def _new_state(params, prompt_ids, output_ids):
            return _TableReqState(output_ids=output_ids)

        process_dict_updates(self.req_states, batch_update, _new_state)

        for st in self.req_states.values():
            out = st.output_ids
            if not out:
                continue

            view = _strip_trailing_negative_token_ids(out)
            if not view:
                continue

            if self.class_token_ids:
                last_class_pos = -1
                for i in range(len(view) - 1, -1, -1):
                    if view[i] in self.class_token_ids:
                        last_class_pos = i
                        break
                if last_class_pos != -1 and last_class_pos != st._last_class_pos:
                    st._last_class_pos = last_class_pos
                    st.expecting_end_coords = False

            if st.insertion_active:
                continue

            if len(view) >= 2 and view[-2] in self.x_token_ids and view[-1] in self.y_token_ids:
                if not st.expecting_end_coords:
                    st.insertion_active = True
                    st.insertion_pos = 0
                    st.expecting_end_coords = True
                else:
                    pass

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.req_states:
            return logits

        for req_idx, st in self.req_states.items():
            if not st.insertion_active:
                continue
            pos = st.insertion_pos
            if pos >= len(self.table_prefix_ids):
                st.insertion_active = False
                continue

            forced_tid = self.table_prefix_ids[pos]
            logits[req_idx].fill_(-float("inf"))
            logits[req_idx, forced_tid] = 0.0

            st.insertion_pos = pos + 1
            if st.insertion_pos >= len(self.table_prefix_ids):
                st.insertion_active = False

        return logits


@dataclass
class _RepReqState:
    output_ids: List[int]
    in_cooldown: bool = False
    segment_start: int = 0
    _last_class_pos: int = -1


class NemotronParseRepetitionStopLogitsProcessor(LogitsProcessor):
    """Force an <x_...> token when consecutive repetition exceeds a threshold."""

    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool) -> None:
        self.enabled = True
        self.req_states: Dict[int, _RepReqState] = {}

        self.max_repetitions = _env_int("NEMOTRON_PARSE_REP_MAX", 10)
        self.ngram_sizes = _env_csv_ints("NEMOTRON_PARSE_REP_NGRAMS", [1, 2, 3, 4, 5])
        max_ngram = max((n for n in self.ngram_sizes if n > 0), default=1)
        default_window = 2 * max_ngram * (self.max_repetitions + 1)
        self.window_size = _env_int("NEMOTRON_PARSE_REP_WINDOW", default_window)

        self.x_token_ids: Set[int] = set()
        self.class_token_ids: Set[int] = set()

        if self.enabled:
            _warn_if_unexpected_arch(vllm_config)
            tok = _load_hf_tokenizer(vllm_config)
            x_ids, _, class_ids = _build_token_sets(tok)
            self.x_token_ids = x_ids
            self.class_token_ids = class_ids

        if not self.x_token_ids:
            self.enabled = False

    @classmethod
    def validate_params(cls, sampling_params):
        return None

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if not self.enabled:
            return

        def _new_state(params, prompt_ids, output_ids):
            return _RepReqState(output_ids=output_ids)

        process_dict_updates(self.req_states, batch_update, _new_state)

        for st in self.req_states.values():
            out = st.output_ids
            if not out:
                continue
            view = _strip_trailing_negative_token_ids(out)
            if not view:
                continue
            if self.class_token_ids:
                last_class_pos = -1
                for i in range(len(view) - 1, -1, -1):
                    if view[i] in self.class_token_ids:
                        last_class_pos = i
                        break
                if last_class_pos != -1 and last_class_pos != st._last_class_pos:
                    st._last_class_pos = last_class_pos
                    st.in_cooldown = False
                    st.segment_start = last_class_pos + 1

    @staticmethod
    def _max_consecutive_repetitions(seq: List[int], n: int) -> int:
        if len(seq) < n:
            return 0
        max_consec = 1
        cur = 1
        prev = tuple(seq[0:n])
        i = n
        while i <= len(seq) - n:
            cur_ng = tuple(seq[i:i + n])
            if cur_ng == prev:
                cur += 1
                if cur > max_consec:
                    max_consec = cur
                i += n
            else:
                cur = 1
                prev = cur_ng
                i += 1
        return max_consec

    def _has_excessive_repetition(self, seq: List[int]) -> bool:
        if not seq:
            return False
        check_seq = seq[-self.window_size:] if len(seq) > self.window_size else seq
        for n in self.ngram_sizes:
            if n <= 0:
                continue
            if self._max_consecutive_repetitions(check_seq, n) > self.max_repetitions:
                return True
        return False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.enabled or not self.req_states:
            return logits

        for req_idx, st in self.req_states.items():
            if st.in_cooldown:
                continue
            view = _strip_trailing_negative_token_ids(st.output_ids)
            seg = view[st.segment_start:]
            if not self._has_excessive_repetition(seg):
                continue

            st.in_cooldown = True
            row = logits[req_idx]
            original = row.clone()
            row.fill_(-float("inf"))
            for tid in self.x_token_ids:
                row[tid] = original[tid]

        return logits
