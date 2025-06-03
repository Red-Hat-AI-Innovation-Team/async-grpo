"""
Reward function adapters registry.

Defines adapters mapping simple string keys to functions that compute
a {"reward": float, "reward_info": {...}} dict given a sample.
"""

from typing import Dict, Callable, Any
from enum import Enum

# from deepscaler_math_utils import extract_answer, grade_answer_mathd, grade_answer_sympy
from countdown_reward import format_reward_function, answer_reward_function


def _extract_reference_and_answer(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts 'parsed_gt_answer' and 'parsed_attempt' fields into the sample dict
    by splitting sample['sample_text'] on sample['input'].
    """
    from deepscaler_math_utils import extract_answer
    original_input = sample['input']
    output = sample['sample_text'].split(original_input)[1]
    # Ground truth answer
    if "\\boxed" in sample.get('answer', ''):
        parsed_gt = extract_answer(sample['answer'])
    else:
        parsed_gt = sample.get('answer')
    # Model attempt
    try:
        parsed_attempt = extract_answer(output)
    except Exception:
        parsed_attempt = ''
    # Annotate sample
    sample['parsed_gt_answer'] = parsed_gt
    sample['parsed_attempt'] = parsed_attempt or ''
    return sample


def mathd_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using mathd (string match) from deepscaler_math_utils.
    Returns a dict with 'reward' (1.0 or 0.0) and 'reward_success'.
    """
    from deepscaler_math_utils import grade_answer_mathd
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_mathd(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def sympy_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Grades the sample using sympy-based checker from deepscaler_math_utils.
    Returns a dict with 'reward' and 'reward_success'.
    """
    from deepscaler_math_utils import grade_answer_sympy
    sample = _extract_reference_and_answer(sample)
    correct = grade_answer_sympy(sample['parsed_attempt'], sample['parsed_gt_answer'])
    reward = float(correct)
    return {"reward": reward}


def countdown_adapter(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adapter for the Countdown Tasks reward.
    """
    RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"
    # Isolate model's generated text by splitting off the prompt
    full_text = sample.get('sample_text', '')
    output = full_text.split(sample.get('input', ''), 1)[1]
    response = "<think>" + output
    # Prepend the RESPONSE_PROMPT to reconstruct the opening <think> tag
    format_r = format_reward_function(response, end_token=sample.get('end_token', ''))
    answer_r = answer_reward_function(response, numbers=sample.get('nums'), target=sample.get('target'))
    reward = format_r * 0.1 + answer_r
    return {"reward": reward, "format_reward": format_r}


def ttrl_reward_function(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adapter using TTRL utils: extract answer via ttrl_utils and grade against majority-vote stored in sample['maj_vote_answer'].
    """
    from ttrl_utils.ttrl_grade import grade_answer
    # extract the attempted answer
    parsed_attempt = sample.get('parsed_attempt', '')

    # fetch majority-voted answer from sample
    maj_answer = sample.get('maj_vote_answer', '')

    # grade the attempt against the majority vote
    correct = grade_answer(parsed_attempt, maj_answer)

    parsed_gt_answer = sample.get('parsed_gt_answer', '')
    ground_truth_correct = grade_answer(parsed_attempt, parsed_gt_answer)

    return {"reward": float(correct), "true_reward": float(ground_truth_correct)}

def ttrl_extract_answer(sample: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Extract answer via ttrl_utils.
    """
    from ttrl_utils.ttrl_parsing import extract_answer as ttrl_extract_answer_
    full_text = sample.get('sample_text', '')
    input_text = sample.get('input', '')
    output = full_text.split(input_text, 1)[1] if input_text in full_text else full_text
    parsed_attempt = ttrl_extract_answer_(output, data_name="math")
    parsed_gt_answer = ttrl_extract_answer_(sample.get('answer', ''), data_name="math")
    return {"parsed_attempt": parsed_attempt, "parsed_gt_answer": parsed_gt_answer}

class RewardType(str, Enum):
    """Enum of available reward adapter names."""
    MATHD = "mathd"
    SYMPY = "sympy"
    COUNTDOWN = "countdown"
    TTRL_REWARD = "ttrl_reward"
    TTRL_EXTRACT_ANSWER = "ttrl_extract_answer"


REWARD_ADAPTERS: Dict[RewardType, Callable[..., Dict[str, Any]]] = {
    RewardType.MATHD: mathd_adapter,
    RewardType.SYMPY: sympy_adapter,
    RewardType.COUNTDOWN: countdown_adapter,
    RewardType.TTRL_REWARD: ttrl_reward_function,
    RewardType.TTRL_EXTRACT_ANSWER: ttrl_extract_answer,
}


def get_reward_adapter(name: RewardType) -> Callable[..., Dict[str, Any]]:
    """
    Look up and return the reward adapter function by RewardType.
    Raises ValueError if not found.
    """
    try:
        return REWARD_ADAPTERS[name]
    except KeyError as e:
        raise ValueError(f"Unknown reward adapter: {name}") from e 