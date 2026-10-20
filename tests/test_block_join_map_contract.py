"""Contract tests for block join/map output parsing.

Run inside the Kalypso repository/virtualenv after copying the module:
    pytest -q tests/test_block_join_map_contract.py
"""

import pytest

from vllm.kalypso.sem_ops.block_join_map import (
    BlockValidationError,
    parse_json_object,
    validate_result_map,
)


def test_parser_accepts_fence_trailing_comma_and_explanation():
    raw = '''```json
    {"R00000000": "mapped text", "R00000001": "another value",}
    ```
    Explanation: both rows were processed.
    '''
    assert parse_json_object(raw) == {
        "R00000000": "mapped text",
        "R00000001": "another value",
    }


def test_join_requires_boolean_result_for_every_pair():
    raw = '{"L00000000_R00000000": true, "L00000000_R00000001": "FALSE",}'
    assert validate_result_map(
        raw,
        ["L00000000_R00000000", "L00000000_R00000001"],
        "boolean",
    ) == {
        "L00000000_R00000000": True,
        "L00000000_R00000001": False,
    }


def test_map_requires_string_result_for_every_row():
    assert validate_result_map(
        '{"R00000000": "A concise summary."}',
        ["R00000000"],
        "text",
    ) == {"R00000000": "A concise summary."}


@pytest.mark.parametrize(
    "raw,expected,kind",
    [
        ('{"R0": "x"}', ["R0", "R1"], "text"),  # missing ID
        ('{"R0": "x", "R1": "y", "R2": "z"}', ["R0", "R1"], "text"),  # invented ID
        ('{"R0": "x", "R0": "y"}', ["R0"], "text"),  # duplicate key
        ('{"R0": 17}', ["R0"], "text"),  # wrong type
        ('{"R0": "yes"}', ["R0"], "boolean"),  # non-Boolean verdict
        ('{"R0": true', ["R0"], "boolean"),  # truncated JSON
    ],
)
def test_invalid_outputs_are_rejected(raw, expected, kind):
    with pytest.raises(BlockValidationError):
        validate_result_map(raw, expected, kind)


def test_block_map_recovers_by_splitting_an_incomplete_block():
    import asyncio
    import json
    from types import SimpleNamespace

    from vllm.kalypso.context import ExecutionState, SemContext, SemanticInput
    from vllm.kalypso.sem_ops.block_join_map import BlockMap, _BlockRow

    class FakeExecutor:
        def __init__(self):
            self.calls = 0

        async def execute(self, raw_request, prompt, max_tokens, pin=False, priority=0):
            self.calls += 1
            content = prompt[-1]["content"]
            marker = "INPUT RECORDS (JSON):\n"
            start = content.index(marker) + len(marker)
            end = content.index("\n\nOPERATION:", start)
            records = json.loads(content[start:end])
            # Simulate an incomplete response for a multi-row block, then valid
            # complete responses once recovery has split it into singleton rows.
            if len(records) > 1:
                text = json.dumps({records[0]["row_id"]: "only first row"})
            else:
                text = json.dumps({records[0]["row_id"]: f"mapped {records[0]['row_id']}"})
            return SimpleNamespace(text=text, finish_reason="stop", request_id="fake")

    async def run():
        executor = FakeExecutor()
        def context(text):
            return SemContext(
                input=SemanticInput(data=[{"role": "user", "type": "text", "content": text}]),
                state=ExecutionState(raw_request=None, executor=executor),
            )

        rows = [
            _BlockRow("R0", "first", 0, ctx=context("first")),
            _BlockRow("R1", "second", 1, ctx=context("second")),
        ]
        op = BlockMap("map each row", max_retries=0, max_tokens=128)
        result = await op._execute_with_recovery(rows)
        assert result == {"R0": "mapped R0", "R1": "mapped R1"}
        assert executor.calls == 3
        assert op.stats["block_splits"] == 1
        assert op.stats["validation_failures"] == 1

    asyncio.run(run())
