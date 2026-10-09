# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
from types import SimpleNamespace

import pytest
from unitree_lerobot import r1a7_rtc_language as language


def args(**changes):
    values = dict(task_before=language.DEFAULT_TASK_BEFORE,
                  task_after=language.DEFAULT_TASK_AFTER,
                  allow_task_override=False, reference_task_before=None)
    return SimpleNamespace(**(values | changes))


TRAINED = {language.DEFAULT_TASK_BEFORE, language.DEFAULT_TASK_AFTER}


def test_original_prompts_keep_original_reference():
    contract = language.resolve_task_contract(args(), TRAINED)
    assert contract['prompts_match_training']
    assert contract['reference_task_before'] == language.DEFAULT_TASK_BEFORE


def test_novel_prompt_requires_explicit_opt_in():
    with pytest.raises(ValueError, match='allow-task-override'):
        language.resolve_task_contract(args(task_before='Bring it to the scanner.'), TRAINED)


@pytest.mark.parametrize('prompt', ['Bring it to the scanner.', '将商品送到扫码器前，等待扫码成功。'])
def test_override_preserves_trained_return_reference(prompt):
    contract = language.resolve_task_contract(args(task_before=prompt, allow_task_override=True), TRAINED)
    assert contract['task_before'] == prompt
    assert not contract['prompts_match_training']
    assert contract['reference_task_before'] == language.DEFAULT_TASK_BEFORE


def test_invalid_reference_still_rejected():
    with pytest.raises(ValueError, match='Reference task'):
        language.resolve_task_contract(args(allow_task_override=True, reference_task_before='Unknown'), TRAINED)


@pytest.mark.parametrize('field', ['task_before', 'task_after'])
def test_empty_prompt_rejected(field):
    with pytest.raises(ValueError, match='empty'):
        language.resolve_task_contract(args(**{field: ' ', 'allow_task_override': True}), TRAINED)


@pytest.mark.parametrize('feedback_fault', [False, True])
def test_independent_entry_dispatch_and_disarm(monkeypatch, tmp_path, feedback_fault):
    import test_r1a7_rtc_live as baseline
    import json
    monkeypatch.setattr(baseline, 'live', language)
    monkeypatch.setattr(language, 'read_training_tasks', lambda *_: TRAINED)
    baseline.test_mock_main_dispatch_or_fault_always_disarms(
        monkeypatch, tmp_path, feedback_fault, False)
    rows = [json.loads(row) for row in (tmp_path / 'mock.jsonl').read_text().splitlines()]
    contract = next(row for row in rows if row['event'] == 'rtc_task_contract')
    assert contract['reference_task_before'] == language.DEFAULT_TASK_BEFORE
