import json
import re
from dataclasses import dataclass
from typing import Any

from vllm.kalypso.budget import KVMemoryManager
from vllm.kalypso.context import SemContext
from vllm.kalypso.sem_ops.base import BaseOp, OpBehavior
from vllm.kalypso.sem_ops.prompt_utils import get_system_prompt


class BlockValidationError(ValueError):
    """Raised when an LLM block output violates the block output contract."""


@dataclass(frozen=True)
class BlockRow:
    row_id: str
    ctx: SemContext


@dataclass(frozen=True)
class Block:
    """
    Ordered group of input rows evaluated by one LLM request.
    """

    block_id: int
    rows: tuple[BlockRow, ...]

    @classmethod
    def from_contexts(
        cls,
        block_id: int,
        contexts: list[SemContext],
    ) -> "Block":
        rows = []
        seen_ids = set()

        for ctx in contexts:
            row_id = str(ctx.state.idx)

            if row_id in seen_ids:
                raise BlockValidationError(
                    f"Duplicate row ID {row_id!r} inside block {block_id}"
                )

            seen_ids.add(row_id)
            rows.append(BlockRow(row_id=row_id, ctx=ctx))

        return cls(
            block_id=block_id,
            rows=tuple(rows),
        )

    @property
    def row_ids(self) -> tuple[str, ...]:
        return tuple(row.row_id for row in self.rows)


@dataclass
class BlockTask:
    """
    One scheduled block = one LLM request.
    """

    parent: "BlockFilter"
    block: Block

    def __post_init__(self):
        self.budget = (
            self.parent.estimate_block_tokens(self.block)
            * KVMemoryManager.get_instance().bytes_per_token
        )

    async def __call__(self):
        return await self.parent._run_block(self.block)


class BlockFilter(BaseOp):
    """
    Evaluate multiple rows in one LLM request.

    Input:
        [ctx0, ctx1, ..., ctxB-1]

    Output contract:
        {
            "row_id_0": true,
            "row_id_1": false,
            ...
        }

    The block is valid only when the output contains exactly one
    boolean answer for every expected row ID.
    """

    LOG = True

    def __init__(
        self,
        instruction: str,
        block_size: int = 1,
        max_tokens: int = 128,
        position: int = -1,
    ):
        super().__init__(
            behavior=OpBehavior.BLOCKING,
            position=position,
            predicate=True,
        )

        if block_size <= 0:
            raise ValueError("block_size must be > 0")

        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")

        self.instruction = instruction
        self.block_size = block_size
        self.max_tokens = max_tokens

        self.stats = {
            "blocks": 0,
            "llm_calls": 0,
            "input_rows": 0,
            "validated_rows": 0,
            "passed_rows": 0,
        }

    def _context_text(self, ctx: SemContext) -> str:
        data = ctx.input.data

        if isinstance(data, list):
            parts = []

            for item in data:
                if not isinstance(item, dict):
                    continue

                content = item.get("content")
                if content:
                    parts.append(str(content))

            if parts:
                return "\n".join(parts)

        return str(data)

    def _build_block_prompt(self, block: Block):
        rows = []

        for row in block.rows:
            rows.append(
                f"ROW_ID: {row.row_id}\n"
                f"{self._context_text(row.ctx)}"
            )

        """content = (
            "You are evaluating a semantic filter over multiple independent rows.\n\n"
            "FILTER CONDITION:\n"
            f"{self.instruction}\n\n"
            "ROWS:\n"
            + "\n\n"
            + "\n\n".join(rows)
            + "\n\n"
            "OUTPUT CONTRACT:\n"
            "Return exactly one JSON object.\n"
            "Every key must be one of the provided ROW_ID values.\n"
            "Every row ID must appear exactly once.\n"
            "Do not omit any row.\n"
            "Do not add any extra row IDs.\n"
            "Every value must be a JSON boolean: true or false.\n"
            "Do not include markdown, explanation, or any other text.\n\n"
            "Example format only:\n"
            '{"0": true, "1": false}\n'
        )"""

        """
        if (len(block.rows) == 1):
            output_contract = (
                "\n\nOUTPUT CONTRACT:\n"
                "Return exactly one JSON object and nothing else.\n"
                "This block contains exactly ONE row.\n"
                "Evaluate only that row.\n"
                "Do not provide an explanation.\n"
                "Do not output any other text.\n"
                "Example format only:\n"
                '{"0": true}\n'
            )
        else:
            valid_ids = ", ".join(repr(row_id) for row_id in block.row_ids)

            output_contract = (
                "\n\nOUTPUT CONTRACT:\n"
                "Return exactly one JSON object and nothing else.\n"
                f"This block contains exactly {len(block.rows)} row(s).\n"
                f"The only valid row IDs are: [{valid_ids}]\n"
                "Use every valid row ID exactly once.\n"
                "Do not add any other row IDs.\n"
                "Do not omit any valid row ID.\n"
                "Every value must be the JSON boolean true or false.\n"
                "Do not include explanations, markdown, comments, or text outside the JSON object.\n"
                "Example format only:\n"
                '{"0": true, "1": false}\n'
            )
        """
        valid_ids = ", ".join(repr(row_id) for row_id in block.row_ids)

        if (len(block.rows) == 1):
            output_contract = (
                "\n\nOUTPUT CONTRACT:\n"
                "Respond STRICTLY with a valid JSON object matching the example format.\n\n"
                f"This block contains exactly {len(block.rows)} row(s).\n"
                f"Allowed Keys: [{valid_ids}]\n"
                f"Allowed Values: boolean (true or false)\n\n"
                "RULES:\n"
                "1. This block contains exactly ONE row. The key is listed in Allowed Keys. Evaluate only that row.\n"
                "2. Do NOT add extra keys, markdown tags (e.g., ```json), or text before/after the JSON.\n\n"
                f"EXAMPLE OUTPUT FORMAT ONLY:\n"
                '{"0": true}\n'
            )
        else:
            output_contract = (
                "\n\nOUTPUT CONTRACT:\n"
                "Respond STRICTLY with a valid JSON object matching the example format.\n\n"
                f"This block contains exactly {len(block.rows)} row(s).\n"
                f"Allowed Keys: [{valid_ids}]\n"
                f"Allowed Values: boolean (true or false)\n\n"
                "RULES:\n"
                "1. You MUST include every key listed in Allowed Keys.\n"
                "2. Do NOT add extra keys, markdown tags (e.g., ```json), or text before/after the JSON.\n\n"
                f"EXAMPLE OUTPUT FORMAT ONLY:\n"
                '{"0": true, "1": false}\n'
            )

        content = (
            "You are evaluating a semantic filter over multiple independent rows.\n\n"
            "FILTER CONDITION:\n"
            f"{self.instruction}\n\n"
            "ROWS:\n"
            + "\n\n".join(rows)
            + output_contract
        )

        return get_system_prompt() + [
            {
                "role": "user",
                "type": "text",
                "content": content,
            }
        ]

    def estimate_block_tokens(self, block: Block) -> int:
        prompt = self._build_block_prompt(block)

        prompt_str = KVMemoryManager.get_instance().apply_chat_template(
            prompt
        )

        prompt_token_len = KVMemoryManager.get_instance().token_length(
            prompt_str
        )

        return prompt_token_len + self.max_tokens

    @staticmethod
    def _remove_code_fence(text: str) -> str:
        text = text.strip()

        if not text.startswith("```"):
            return text

        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines).strip()

    @staticmethod
    def _json_object_hook(pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
        result = {}

        for key, value in pairs:
            if key in result:
                raise BlockValidationError(
                    f"Duplicate row ID in LLM output: {key!r}"
                )

            result[key] = value

        return result

    @classmethod
    def validate_output(
        cls,
        text: str,
        expected_ids: tuple[str, ...],
    ) -> dict[str, bool]:
        cleaned = cls._remove_code_fence(text)

        cleaned = re.sub(
            r",\s*([}\]])",
            r"\1",
            cleaned,
        )

        try:
            parsed = json.loads(
                cleaned,
                object_pairs_hook=cls._json_object_hook,
            )
        except json.JSONDecodeError as exc:
            raise BlockValidationError(
                f"Malformed JSON block output: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise BlockValidationError(
                f"Expected JSON object, got {type(parsed).__name__}"
            )

        actual_ids = set(parsed.keys())
        expected_set = set(expected_ids)

        missing = expected_set - actual_ids
        extra = actual_ids - expected_set

        if missing:
            raise BlockValidationError(
                f"Missing row IDs in block output: {sorted(missing)}"
            )

        if extra:
            raise BlockValidationError(
                f"Invented row IDs in block output: {sorted(extra)}"
            )

        normalized = {}

        for row_id in expected_ids:
            value = parsed[row_id]

            if isinstance(value, bool):
                normalized[row_id] = value
                continue

            if isinstance(value, str):
                lowered = value.strip().lower()

                if lowered == "true":
                    normalized[row_id] = True
                    continue

                if lowered == "false":
                    normalized[row_id] = False
                    continue

            raise BlockValidationError(
                f"Row {row_id!r} has invalid boolean value: {value!r}"
            )

        return normalized

    async def _run_block(self, block: Block):
        if not block.rows:
            raise BlockValidationError(
                f"Block {block.block_id} is empty"
            )

        executor = block.rows[0].ctx.state.executor
        raw_request = block.rows[0].ctx.state.raw_request

        prompt = self._build_block_prompt(block)

        self.stats["llm_calls"] += 1

        output = await executor.execute(
            raw_request=raw_request,
            prompt=prompt,
            max_tokens=self.max_tokens,
            pin=False,
            priority=0,
        )

        """
        decisions = self.validate_output(
            output.text or "",
            block.row_ids,
        )
        """
        raw_text = output.text or ""

        print(
            "[block-filter] RAW "
            f"block={block.block_id} "
            f"row_ids={block.row_ids} "
            f"finish_reason={output.finish_reason!r} "
            f"text={raw_text!r}"
        )

        try:
            decisions = self.validate_output(
                raw_text,
                block.row_ids,
            )
        except BlockValidationError as exc:
            print(
                "[block-filter] VALIDATION FAILED "
                f"block={block.block_id} "
                f"row_ids={block.row_ids} "
                f"error={exc} "
                f"raw={raw_text!r}"
            )
            raise

        self.stats["blocks"] += 1
        #self.stats["input_rows"] += len(block.rows)
        self.stats["validated_rows"] += len(block.rows)

        passed_count = sum(
            1 for row_id in block.row_ids
            if decisions[row_id]
        )
        self.stats["passed_rows"] += passed_count

        if self.LOG:
            print(
                "[block-filter] "
                f"block={block.block_id} "
                f"rows={len(block.rows)} "
                f"request_id={output.request_id} "
                f"validated={len(block.rows)} "
                f"passed={passed_count}"
            )

        return block.block_id, [
            (row.ctx, decisions[row.row_id])
            for row in block.rows
        ]

    async def __call__(self, ctxs, priority: int = 0):
        from vllm.kalypso.execution.pipeline_execution import BlockingExecutor

        if not isinstance(ctxs, list):
            raise TypeError(
                "BlockFilter expects a list of SemContext objects"
            )

        if not ctxs:
            return []

        """
        blocks = [
            Block.from_contexts(
                block_id=block_id,
                contexts=ctxs[start:start + self.block_size],
            )
            for block_id, start in enumerate(
                range(0, len(ctxs), self.block_size)
            )
        ]

        self.stats = {
            "blocks": 0,
            "llm_calls": 0,
            "input_rows": 0,
            "validated_rows": 0,
            "passed_rows": 0,
        }
        """
        self.stats = {
            "blocks": 0,
            "llm_calls": 0,
            "input_rows": 0,
            "validated_rows": 0,
            "passed_rows": 0,
        }

        print(
            f"[block-filter] RECEIVED ctxs={len(ctxs)} "
            f"block_size={self.block_size}"
        )

        blocks = [
            Block.from_contexts(
                block_id=block_id,
                contexts=ctxs[start:start + self.block_size],
            )
            for block_id, start in enumerate(
                range(0, len(ctxs), self.block_size)
            )
        ]

        self.stats["input_rows"] = sum(
            len(block.rows) for block in blocks
        )

        print(
            "[block-filter] CREATED BLOCKS "
            + str([len(block.rows) for block in blocks])
        )

        print(
            f"[block-filter] TOTAL BLOCK ROWS="
            f"{self.stats['input_rows']}"
        )

        results = await BlockingExecutor.execute_tasks(
            seeds=blocks,
            task_builder=lambda block: BlockTask(self, block),
        )

        results.sort(key=lambda item: item[0])

        passed_contexts = []

        for _, block_results in results:
            for ctx, passed in block_results:
                verdict = "true" if passed else "false"

                ctx.output.append({
                    str(self.__class__): verdict
                })

                if passed:
                    passed_contexts.append(ctx)

        complete = (
            self.stats["validated_rows"] == self.stats["input_rows"]
            and self.stats["llm_calls"] == self.stats["blocks"]
        )

        print(
            "[block-filter] "
            f"SUMMARY block_size={self.block_size} "
            f"blocks={self.stats['blocks']} "
            f"llm_calls={self.stats['llm_calls']} "
            f"input_rows={self.stats['input_rows']} "
            f"validated_rows={self.stats['validated_rows']} "
            f"passed_rows={self.stats['passed_rows']} "
            f"complete={complete}"
        )

        if not complete:
            raise BlockValidationError(
                "BlockFilter did not validate a complete result for every input row"
            )

        return passed_contexts
