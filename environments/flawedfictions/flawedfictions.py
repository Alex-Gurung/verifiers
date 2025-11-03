import verifiers as vf

from datasets import load_dataset

# no system prompt used; parser returns raw assistant content
from environments.flawedfictions.ff_reward_func_utils import (
    listify_lines,
    precompute_story_sentences,
    make_binary_rewards,
    make_localization_rewards,
    parse_paper_binary_label,
    is_paper_binary_formatted,
    parse_boxed_binary_label,
    is_boxed_binary_formatted,
)

"""FlawedFictions environment: dataset mapping + rubric wiring.

Rewards are produced via factories in ff_reward_func_utils:
- Binary stage: format check + accuracy check.
- Localization stage: format check + full correctness (gated on binary).
"""


def load_environment(
    system_prompt: str | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    train_source: str = "flawed_fictions",
    eval_source: str = "flawed_fictions",
    prompt_name: str = "simple_boxed_prompt.txt",
    split_ratio: float = 0.8,
    binary_reward_weight: float = 1.0,
    binary_format_reward_weight: float = 0.0,
    line_identification_reward_weight: float = 0.0,
    line_identification_format_reward_weight: float = 0.0,
):
    """
    Args:
        system_prompt: The system prompt to use for the environment.
        num_train_examples: The number of train examples to use. -1 for all
        num_eval_examples: The number of eval examples to use. -1 for all
        train_source: The source of the train dataset, accepts (flawed_fictions [414 datapoints] or flawed_fictions_long [200 datapoints])
        eval_source: The source of the eval dataset, accepts (flawed_fictions [414 datapoints] or flawed_fictions_long [200 datapoints])
        prompt_name: The prompt file to use, accepts (simple_boxed_prompt.txt, conterror_detector_prompt.txt, conterror_detector_prompt_cot.txt)
        split_ratio: The ratio of train to eval examples, default is 0.8. If num_train_examples and num_eval_examples are set or train and eval source are different, this is ignored.
        binary_reward_weight: Weight of the binary reward (Yes/No prediction task).
        binary_format_reward_weight: Weight of the format reward (for the binary task). Depends on the prompt used.
        line_identification_reward_weight: Weight of the line identification reward.
        line_identification_format_reward_weight: Weight of the line identification format reward. Looks for
    We introduce a new format in the `simple_boxed_prompt.txt` prompt, where the answer is expected to be Yes/No inside \\boxed{...} tags. For this prompt we use a simple boxed parser, but for others we default to the format presented in the original paper.
    """

    assert train_source in [
        "flawed_fictions",
        "flawed_fictions_long",
    ], "train_source must be either flawed_fictions or flawed_fictions_long"
    assert eval_source in [
        "flawed_fictions",
        "flawed_fictions_long",
    ], "eval_source must be either flawed_fictions or flawed_fictions_long"
    assert prompt_name in [
        "simple_boxed_prompt.txt",
        "conterror_detector_prompt.txt",
        "conterror_detector_prompt_cot.txt",
    ], (
        "prompt_name must be either simple_boxed_prompt.txt, conterror_detector_prompt.txt, conterror_detector_prompt_cot.txt"
    )
    with open(f"environments/flawedfictions/prompts/{prompt_name}", "r") as f:
        task_prompt = f.read()

    # construct dataset, if train and eval source are the same, we use the same dataset for train and eval
    if train_source == eval_source:
        total_dataset = load_dataset("kahuja/flawed-fictions", split=train_source)
        num_train_examples = (
            int(len(total_dataset) * split_ratio)
            if num_train_examples == -1
            else num_train_examples
        )
        num_eval_examples = (
            len(total_dataset) - num_train_examples
            if num_eval_examples == -1
            else num_eval_examples
        )
        assert num_train_examples + num_eval_examples <= len(total_dataset), (
            "num_train_examples + num_eval_examples must be less than or equal to the total number of examples, or set to -1 to use the default 80/20 split"
        )
        dataset = total_dataset.select(range(num_train_examples))
        eval_dataset = total_dataset.select(
            range(num_train_examples, num_train_examples + num_eval_examples)
        )
    else:
        dataset = load_dataset("kahuja/flawed-fictions", split=train_source)
        eval_dataset = load_dataset("kahuja/flawed-fictions", split=eval_source)
        if num_train_examples != -1:
            dataset = dataset.select(range(num_train_examples))
        if num_eval_examples != -1:
            eval_dataset = eval_dataset.select(range(num_eval_examples))

    def to_answer(x):
        sents_proc = precompute_story_sentences(x["story"])
        return {
            # "prompt": task_prompt.replace("{story}", x["story"]),
            "question": task_prompt.replace("{story}", x["story"]),
            "info": {
                "cont_error": int(x["cont_error"]),
                # strings that introduce the error and those contradicted earlier
                "cont_error_lines": listify_lines(x["cont_error_lines"]),
                "contradicted_lines": listify_lines(x["contradicted_lines"]),
                # precomputed processed sentence view for speed
                "story_sents_proc": sents_proc,
            },
        }

    dataset = dataset.map(to_answer, num_proc=10, remove_columns=dataset.column_names)
    eval_dataset = eval_dataset.map(
        to_answer, num_proc=10, remove_columns=eval_dataset.column_names
    )

    # Choose parser and reward functions based on prompt style
    parser = vf.Parser()  # return raw assistant content for reward parsing

    parse_binary_label_fn = (
        parse_boxed_binary_label
        if prompt_name == "simple_boxed_prompt.txt"
        else parse_paper_binary_label
    )
    is_binary_formatted_fn = (
        is_boxed_binary_formatted
        if prompt_name == "simple_boxed_prompt.txt"
        else is_paper_binary_formatted
    )
    binary_format, binary_acc = make_binary_rewards(
        parse_binary_label_fn, is_binary_formatted_fn
    )
    localization_format, localization_full = make_localization_rewards(
        parse_binary_label_fn, is_binary_formatted_fn
    )

    reward_funcs = [binary_format, binary_acc, localization_format, localization_full]

    reward_weights = [
        float(binary_format_reward_weight),
        float(binary_reward_weight),
        float(line_identification_format_reward_weight),
        float(line_identification_reward_weight),
    ]

    rubric = vf.Rubric(parser=parser, funcs=reward_funcs, weights=reward_weights)

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
