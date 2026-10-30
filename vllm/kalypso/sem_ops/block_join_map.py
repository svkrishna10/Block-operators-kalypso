"""Block-oriented semantic join and map operators for Kalypso.

The operators issue one LLM request for each block. Results are keyed by
stable row/pair IDs and strictly validated. If a response cannot be validated,
retries are attempted and the block is recursively split to isolate the bad
sub-block. Pipeline integration is intentionally left to a later milestone;
these operators run as OpBehavior.BLOCKING operators.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .base import BaseOp, OpBehavior
from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import ExecutionState, SemContext, SemanticInput
from .prompt_utils import get_system_prompt


class BlockValidationError(ValueError):
    """Raised when an LLM response violates the block output contract."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse the first JSON object in a response.

    Accepts Markdown fences, leading/trailing commentary, and trailing commas.
    It does not excuse missing/extra IDs or invalid value types; those are
    checked separately by ``validate_result_map``.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise BlockValidationError("Empty block output")

    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_keys)
    # Try each opening brace so a brief preamble before the JSON is harmless.
    for start, char in enumerate(raw):
        if char != "{":
            continue
        candidate = raw[start:]
        # Models commonly emit a trailing comma before the closing brace.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            value, _end = decoder.raw_decode(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict):
            return value

    raise BlockValidationError(
        f"Malformed block JSON; response did not contain a valid JSON object: {raw[:240]!r}"
    )

"""
def validate_result_map(
    raw: str,
    expected_ids: Iterable[str],
    value_kind: str,
) -> dict[str, Any]:
    
    Validate exact ID coverage and value types for a block response.

    ``value_kind`` is ``"boolean"`` for join results or ``"text"`` for map
    results. String TRUE/FALSE values are normalized for robustness; no other
    coercions are performed.
    
    expected_order = list(expected_ids)
    expected = set(expected_order)
    parsed = parse_json_object(raw)
    actual = set(parsed)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise BlockValidationError(
            f"ID coverage mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    normalized: dict[str, Any] = {}
    for row_id in expected_order:
        value = parsed[row_id]
        if value_kind == "boolean":
            if type(value) is bool:
                normalized[row_id] = value
            elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
                normalized[row_id] = value.strip().lower() == "true"
            else:
                raise BlockValidationError(
                    f"Expected Boolean for ID {row_id!r}; got {type(value).__name__}"
                )
        elif value_kind == "text":
            if not isinstance(value, str):
                raise BlockValidationError(
                    f"Expected string for ID {row_id!r}; got {type(value).__name__}"
                )
            normalized[row_id] = value.strip()
        else:
            raise ValueError(f"Unknown block value_kind: {value_kind!r}")
    return normalized
"""
def validate_result_map(
    raw: str,
    expected_ids: Iterable[str],
    value_kind: str,
) -> dict[str, Any]:
    """Validate complete ID coverage and result types.

    Accepts only unambiguous ID aliases and simple text wrappers.
    Missing rows, unknown IDs, duplicate aliases, and ambiguous values fail.
    """
    expected_order = list(expected_ids)
    expected = set(expected_order)
    parsed = parse_json_object(raw)

    # Canonicalize only known, unambiguous ID variants:
    #   R00000004       -> R00000004
    #   r00000004       -> R00000004
    #   row_id_R00000004 -> R00000004
    expected_by_fold = {row_id.casefold(): row_id for row_id in expected_order}
    canonicalized: dict[str, Any] = {}

    for key, value in parsed.items():
        candidate = key
        if candidate.casefold().startswith("row_id_"):
            candidate = candidate[len("row_id_"):]

        canonical_id = expected_by_fold.get(candidate.casefold())

        if canonical_id is None:
            # Keep unrecognized keys so they will fail the extra-ID check.
            canonicalized[key] = value
            continue

        if canonical_id in canonicalized:
            raise BlockValidationError(
                f"Multiple output keys resolve to ID {canonical_id!r}"
            )

        canonicalized[canonical_id] = value

    parsed = canonicalized
    actual = set(parsed)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise BlockValidationError(
            f"ID coverage mismatch; missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    normalized: dict[str, Any] = {}

    for row_id in expected_order:
        value = parsed[row_id]

        if value_kind == "boolean":
            if type(value) is bool:
                normalized[row_id] = value
            elif (
                isinstance(value, str)
                and value.strip().lower() in {"true", "false"}
            ):
                normalized[row_id] = value.strip().lower() == "true"
            else:
                raise BlockValidationError(
                    f"Expected Boolean for ID {row_id!r}; "
                    f"got {type(value).__name__}"
                )

        elif value_kind == "text":
            # Accept a single, clearly labeled text wrapper, such as
            # {"summary": "..."}, or a one-element list containing text.
            if isinstance(value, dict) and len(value) == 1:
                wrapper_key, wrapper_value = next(iter(value.items()))
                if (
                    wrapper_key.casefold()
                    in {"text", "summary", "output", "result", "answer"}
                    and isinstance(wrapper_value, str)
                ):
                    value = wrapper_value

            if (
                isinstance(value, list)
                and len(value) == 1
                and isinstance(value[0], str)
            ):
                value = value[0]

            if not isinstance(value, str):
                raise BlockValidationError(
                    f"Expected string for ID {row_id!r}; "
                    f"got {type(value).__name__}"
                )

            normalized[row_id] = value.strip()

        else:
            raise ValueError(f"Unknown block value_kind: {value_kind!r}")

    return normalized

@dataclass(frozen=True)
class _BlockRow:
    row_id: str
    payload: str
    order: int
    ctx: SemContext | None = None
    left_ctx: SemContext | None = None
    right_ctx: SemContext | None = None


@dataclass(frozen=True)
class _BlockBatch:
    batch_index: int
    rows: list[_BlockRow]


class _BlockOperatorBase(BaseOp):
    """Shared prompt construction, validation, memory accounting and recovery."""

    LOG = True

    def __init__(
        self,
        instruction: str,
        *,
        block_size: int = 4,
        max_tokens: int = 512,
        max_retries: int = 1,
        tokens_per_row: int = 12,
        position: int = -1,
    ):
        super().__init__(behavior=OpBehavior.BLOCKING, position=position, predicate=False)
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.instruction = instruction
        self.block_size = block_size
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.tokens_per_row = max(1, tokens_per_row)
        self.stats = {
            "llm_calls": 0,
            "validation_failures": 0,
            "recovery_calls": 0,
            "block_splits": 0,
            "rows_processed": 0,
            "blocks_processed": 0,
        }

    @staticmethod
    def _context_text(ctx: SemContext) -> str:
        """Extract user/assistant content while excluding repeated system text."""
        chunks: list[str] = []
        for message in getattr(ctx.input, "data", None) or []:
            if not isinstance(message, dict):
                chunks.append(str(message))
                continue
            if str(message.get("role", "")).lower() == "system":
                continue
            content = message.get("content", "")
            if content is None:
                continue
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            elif not isinstance(content, str):
                chunks.append(json.dumps(content, ensure_ascii=False, default=str))
        for message in getattr(ctx.input, "right_data", None) or []:
            if not isinstance(message, dict):
                chunks.append(str(message))
                continue
            if str(message.get("role", "")).lower() == "system":
                continue
            content = message.get("content", "")
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            elif content is not None and not isinstance(content, str):
                chunks.append(json.dumps(content, ensure_ascii=False, default=str))
        return "\n\n".join(chunks).strip()

    def _records(self, rows: list[_BlockRow]) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _value_kind(self) -> str:
        raise NotImplementedError

    def _operation_label(self) -> str:
        raise NotImplementedError

    def _build_prompt(self, rows: list[_BlockRow]) -> list[dict[str, Any]]:
        records = self._records(rows)
        contract = (
            "Treat every record independently. Return exactly one JSON object, "
            "with every supplied ID as a key and no other keys. Do not use Markdown, "
            "code fences, comments, or explanatory text."
        )

        if self._value_kind() == "boolean":
            example = {
                row.row_id: (idx % 2 == 0)
                for idx, row in enumerate(rows)
            }
        else:
            example = {
                row.row_id: f"<mapped text for {row.row_id}>"
                for row in rows
            }

        if self._value_kind() == "boolean":
            contract += " Every value must be the JSON boolean true or false."
        else:
            contract += " Every value must be a JSON string containing that row's mapped output."

        # Keep the data before the instruction, matching Kalypso's existing
        # data-first prompt layout and preserving the opportunity for prefix reuse.
        """
        user_content = (
            f"INPUT RECORDS (JSON):\n{json.dumps(records, ensure_ascii=False)}\n\n"
            f"OPERATION: {self._operation_label()}\n"
            f"INSTRUCTION: {self.instruction}\n\n"
            f"OUTPUT CONTRACT: {contract}\n"
            "Return only the JSON object."
        )
        """
        user_content = (
            f"INPUT RECORDS (JSON):\n"
            f"{json.dumps(records, ensure_ascii=False)}\n\n"
            f"OPERATION: {self._operation_label()}\n"
            f"INSTRUCTION: {self.instruction}\n\n"
            f"OUTPUT CONTRACT: {contract}\n\n"
            "OUTPUT SHAPE EXAMPLE (illustrative values only):\n"
            f"{json.dumps(example, ensure_ascii=False)}\n\n"
            "Use the exact IDs shown in the example. Replace the example "
            "values with the actual results for the input records. "
            "Return only one valid JSON object. Do not add Markdown, "
            "comments, or explanatory text."
        )
        
        return get_system_prompt() + [{
            "role": "user",
            "type": "text",
            "content": user_content,
        }]

    def _planned_max_tokens(self, rows: list[_BlockRow], retry_attempt: int = 0) -> int:
        # Larger blocks need more output space. A retry gets a larger completion
        # allowance (when the configured cap permits it), helping recover from
        # truncation before splitting the block.
        base = max(32, self.tokens_per_row * len(rows))
        return min(self.max_tokens, base * (2 ** retry_attempt))

    def _estimate_budget(self, rows: list[_BlockRow]) -> int:
        manager = KVMemoryManager.get_instance()
        prompt = self._build_prompt(rows)
        prompt_text = manager.apply_chat_template(prompt)
        prompt_tokens = manager.token_length(prompt_text)
        # Reserve for the largest completion allowance used by the configured
        # retry policy, avoiding both under-accounting and needless reservation
        # of the full cap when normal outputs are short.
        total_tokens = prompt_tokens + self._planned_max_tokens(rows, self.max_retries)
        return max(manager.bytes_per_token, total_tokens * manager.bytes_per_token)

    def _structured_outputs_for_rows(
        self, rows: list[_BlockRow]
    ) -> dict[str, Any]:
        """Constrain generation to this batch's exact result IDs and value types."""
        kind = self._value_kind()
        if kind == "boolean":
            value_schema = {"type": "boolean"}
        elif kind == "text":
            value_schema = {"type": "string"}
        else:
            raise ValueError(f"Unknown block value_kind: {kind!r}")

        expected_ids = [row.row_id for row in rows]
        schema = {
            "type": "object",
            "properties": {
                row_id: dict(value_schema) for row_id in expected_ids
            },
            "required": expected_ids,
            "additionalProperties": False,
        }
        return {"json": schema}

    async def _execute_once(
        self,
        rows: list[_BlockRow],
        retry_attempt: int = 0,
    ) -> dict[str, Any]:
        if not rows:
            return {}
        prompt = self._build_prompt(rows)
        executor_ctx = next(
            (row.ctx or row.left_ctx for row in rows if row.ctx is not None or row.left_ctx is not None),
            None,
        )
        if executor_ctx is None:
            raise RuntimeError("A block row is missing its source context")

        self.stats["llm_calls"] += 1
        output = await executor_ctx.state.executor.execute(
            raw_request=executor_ctx.state.raw_request,
            prompt=prompt,
            max_tokens=self._planned_max_tokens(rows, retry_attempt),
            pin=False,
            priority=0,
            structured_outputs=self._structured_outputs_for_rows(rows),
        )
        text = getattr(output, "text", "") or ""
        try:
            return validate_result_map(
                text,
                (row.row_id for row in rows),
                self._value_kind(),
            )
        except BlockValidationError:
            self.stats["validation_failures"] += 1
            raise

    @staticmethod
    def _is_context_overflow(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(term in message for term in (
            "context length", "max_tokens", "too large", "maximum context", "context window"
        ))

    async def _execute_with_recovery(self, rows: list[_BlockRow]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.stats["recovery_calls"] += 1
            try:
                return await self._execute_once(rows, retry_attempt=attempt)
            except BlockValidationError as exc:
                last_error = exc
            except Exception as exc:
                # Only a context-window failure is recoverable by splitting.
                if not self._is_context_overflow(exc) or len(rows) == 1:
                    raise
                last_error = exc
                break

        if len(rows) == 1:
            assert last_error is not None
            raise BlockValidationError(
                f"Single-row block still invalid after {self.max_retries + 1} attempt(s): {last_error}"
            ) from last_error

        self.stats["block_splits"] += 1
        midpoint = len(rows) // 2
        if self.LOG:
            print(
                f"[{self.__class__.__name__}] RECOVERY split "
                f"rows={len(rows)} into {midpoint}+{len(rows) - midpoint}; reason={last_error}"
            )
        # Sequential recursive recovery avoids multiplying KV pressure while the
        # parent block's budget is still reserved by BlockingExecutor.
        left_results = await self._execute_with_recovery(rows[:midpoint])
        right_results = await self._execute_with_recovery(rows[midpoint:])
        return {**left_results, **right_results}

    async def _run_batches(self, batches: list[_BlockBatch]) -> list[tuple[int, dict[str, Any]]]:
        if not batches:
            return []

        # BlockingExecutor waits until a task's memory budget can be admitted.
        # Split over-budget requests before queueing them so a batch larger than
        # the entire KV budget cannot wait forever for capacity that will never
        # become available.
        manager = KVMemoryManager.get_instance()
        capacity = manager.capacity()
        prepared: list[_BlockBatch] = []

        def fit_batch(batch: _BlockBatch) -> None:
            budget = self._estimate_budget(batch.rows)
            if budget <= capacity:
                prepared.append(_BlockBatch(len(prepared), batch.rows))
                return
            if len(batch.rows) == 1:
                raise MemoryError(
                    f"Block request for row {batch.rows[0].row_id!r} estimates "
                    f"{budget:,} KV bytes, exceeding manager capacity {capacity:,}. "
                    "Reduce max_tokens or the input row size."
                )
            self.stats["block_splits"] += 1
            midpoint = len(batch.rows) // 2
            fit_batch(_BlockBatch(batch.batch_index, batch.rows[:midpoint]))
            fit_batch(_BlockBatch(batch.batch_index, batch.rows[midpoint:]))

        for batch in batches:
            fit_batch(batch)

        parent = self

        def build_task(batch: _BlockBatch):
            class BlockTask:
                def __init__(self):
                    self.budget = parent._estimate_budget(batch.rows)

                async def __call__(self):
                    parent.stats["blocks_processed"] += 1
                    try:
                        values = await parent._execute_with_recovery(batch.rows)
                        return batch.batch_index, values, None
                    except Exception as exc:
                        # Return failures as values so an exception cannot kill a
                        # BlockingExecutor worker and strand its queue.
                        return batch.batch_index, None, exc

            return BlockTask()

        
        from vllm.kalypso.execution.pipeline_execution import BlockingExecutor
        raw_results = await BlockingExecutor.execute_tasks(
            seeds=prepared,
            task_builder=build_task,
        )
        failures = [item[2] for item in raw_results if item[2] is not None]
        if failures:
            raise failures[0]
        results = [(item[0], item[1]) for item in raw_results]
        return sorted(results, key=lambda item: item[0])


class BlockMap(_BlockOperatorBase):
    """Map several input tuples in one LLM request, returning one string per ID."""

    def __init__(
        self,
        instruction: str,
        *,
        block_size: int = 4,
        max_tokens: int = 512,
        max_retries: int = 1,
        tokens_per_row: int = 32,
        position: int = -1,
    ):
        super().__init__(
            instruction,
            block_size=block_size,
            max_tokens=max_tokens,
            max_retries=max_retries,
            tokens_per_row=tokens_per_row,
            position=position,
        )

    def _records(self, rows: list[_BlockRow]) -> list[dict[str, Any]]:
        return [{"row_id": row.row_id, "context": row.payload} for row in rows]

    def _value_kind(self) -> str:
        return "text"

    def _operation_label(self) -> str:
        return "BLOCK SEMANTIC MAP"

    async def __call__(self, ctxs: list[SemContext]) -> list[SemContext]:
        if not ctxs:
            return []
        rows = [
            _BlockRow(
                row_id=f"R{idx:08d}",
                payload=self._context_text(ctx),
                order=idx,
                ctx=ctx,
            )
            for idx, ctx in enumerate(ctxs)
        ]
        batches = [
            _BlockBatch(i, rows[start:start + self.block_size])
            for i, start in enumerate(range(0, len(rows), self.block_size))
        ]
        batch_results = await self._run_batches(batches)
        by_id: dict[str, str] = {}
        for _batch_index, values in batch_results:
            by_id.update(values)

        expected = {row.row_id for row in rows}
        if set(by_id) != expected:
            raise BlockValidationError(
                f"BlockMap output coverage mismatch after recovery; "
                f"missing={sorted(expected - set(by_id))[:8]}"
            )

        out: list[SemContext] = []
        for row in rows:
            assert row.ctx is not None
            mapped = by_id[row.row_id]
            row.ctx.input.data = list(row.ctx.input.data or []) + [{
                "role": "assistant",
                "type": "text",
                "content": mapped,
            }]
            row.ctx.input.right_data = []
            row.ctx.output.append({str(self.__class__): mapped})
            out.append(row.ctx)
        self.stats["rows_processed"] += len(out)
        if self.LOG:
            print(f"[BlockMap] stats={self.stats}")
        return out


class BlockJoin(_BlockOperatorBase):
    """Batched semantic join over k left rows x j right rows.

    ``left_block_size=1`` yields 1×j requests; values greater than one yield
    k×j requests. Only pairs classified TRUE are emitted downstream.
    """

    def __init__(
        self,
        instruction: str,
        right_table: list[SemContext],
        *,
        left_block_size: int = 1,
        right_block_size: int = 4,
        max_tokens: int = 512,
        max_retries: int = 1,
        tokens_per_row: int = 8,
        position: int = -1,
    ):
        super().__init__(
            instruction,
            # The per-request batch size is set by the k×j rectangle below,
            # not by a flat row chunk size.
            block_size=1,
            max_tokens=max_tokens,
            max_retries=max_retries,
            tokens_per_row=tokens_per_row,
            position=position,
        )
        if left_block_size < 1 or right_block_size < 1:
            raise ValueError("left_block_size and right_block_size must be >= 1")
        self.right_table = list(right_table or [])
        self.left_block_size = left_block_size
        self.right_block_size = right_block_size

    def _records(self, rows: list[_BlockRow]) -> list[dict[str, Any]]:
        return [{
            "pair_id": row.row_id,
            "left": self._context_text(row.left_ctx),
            "right": self._context_text(row.right_ctx),
        } for row in rows]

    def _value_kind(self) -> str:
        return "boolean"

    def _operation_label(self) -> str:
        return "BLOCK SEMANTIC JOIN"

    async def __call__(self, left_ctxs: list[SemContext]) -> list[SemContext]:
        if not left_ctxs or not self.right_table:
            return []

        batches: list[_BlockBatch] = []
        batch_index = 0
        pair_rows: dict[str, tuple[int, int, SemContext, SemContext]] = {}
        for left_start in range(0, len(left_ctxs), self.left_block_size):
            left_chunk = list(enumerate(
                left_ctxs[left_start:left_start + self.left_block_size],
                start=left_start,
            ))
            for right_start in range(0, len(self.right_table), self.right_block_size):
                right_chunk = list(enumerate(
                    self.right_table[right_start:right_start + self.right_block_size],
                    start=right_start,
                ))
                rows: list[_BlockRow] = []
                for left_idx, left_ctx in left_chunk:
                    for right_idx, right_ctx in right_chunk:
                        pair_id = f"L{left_idx:08d}_R{right_idx:08d}"
                        row = _BlockRow(
                            row_id=pair_id,
                            payload="",
                            order=left_idx * len(self.right_table) + right_idx,
                            left_ctx=left_ctx,
                            right_ctx=right_ctx,
                        )
                        rows.append(row)
                        pair_rows[pair_id] = (left_idx, right_idx, left_ctx, right_ctx)
                batches.append(_BlockBatch(batch_index, rows))
                batch_index += 1

        batch_results = await self._run_batches(batches)
        decisions: dict[str, bool] = {}
        for _batch_index, values in batch_results:
            decisions.update(values)

        expected = set(pair_rows)
        if set(decisions) != expected:
            raise BlockValidationError(
                f"BlockJoin output coverage mismatch after recovery; "
                f"missing={sorted(expected - set(decisions))[:8]}"
            )

        matches: list[tuple[int, int, SemContext]] = []
        for pair_id, passed in decisions.items():
            if not passed:
                continue
            left_idx, right_idx, left_ctx, right_ctx = pair_rows[pair_id]
            left_messages = list(left_ctx.input.data or []) + list(
                getattr(left_ctx.input, "right_data", None) or []
            )
            right_messages = list(right_ctx.input.data or []) + list(
                getattr(right_ctx.input, "right_data", None) or []
            )
            left_has_system = any(
                isinstance(message, dict)
                and str(message.get("role", "")).lower() == "system"
                for message in left_messages
            )
            joined_data = left_messages + [
                message
                for message in right_messages
                if not (
                    left_has_system
                    and isinstance(message, dict)
                    and str(message.get("role", "")).lower() == "system"
                )
            ]
            joined_input = SemanticInput(data=joined_data, right_data=[])
            joined_ctx = SemContext(
                input=joined_input,
                output=[],
                state=ExecutionState(
                    raw_request=left_ctx.state.raw_request,
                    pin_req_id=None,
                    executor=left_ctx.state.executor,
                    predicate=left_ctx.state.predicate,
                    helper_score=None,
                    idx=left_ctx.state.idx,
                    stage_id=left_ctx.state.stage_id,
                ),
            )
            joined_ctx.output.append({
                str(self.__class__): {
                    "left_row_id": left_idx,
                    "right_row_id": right_idx,
                    "matched": True,
                }
            })
            matches.append((left_idx, right_idx, joined_ctx))

        matches.sort(key=lambda item: (item[0], item[1]))
        out = [ctx for _left_idx, _right_idx, ctx in matches]
        self.stats["rows_processed"] += len(pair_rows)
        if self.LOG:
            print(
                f"[BlockJoin] candidate_pairs={len(pair_rows)} matched_pairs={len(out)} "
                f"left_block_size={self.left_block_size} "
                f"right_block_size={self.right_block_size} stats={self.stats}"
            )
        return out
