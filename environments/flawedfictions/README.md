# flawedfictions

### Overview
- **Environment ID**: `flawedfictions`
- **Short description**: Single-turn plot-hole detection task in short stories, proposed by Ahuja et al., 2025. Includes binary prediction task and line identification task.
- **Tags**: single-turn, story-understanding

### Datasets
- **Primary dataset(s)**: `kahuja/flawed-fictions`, short or long splits.
- **Source links**: https://huggingface.co/datasets/kahuja/flawed-fictions
- **Split sizes**: By default 331/83 (80/20 train eval split on flawed_fictions subset).

### Task
- **Type**: single-turn
- **Parser**: custom
- **Rubric overview**: Binary reward for plot-hole detection (accuracy), optional line identification and format rewards 

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval flawedfictions
```

Configure model and sampling:

```bash
uv run vf-eval flawedfictions   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Number of train examples (use -1 for all) |
| `num_eval_examples` | int | `-1` | Number of eval examples (use -1 for all) |
| `train_source` | str | `"flawed_fictions"` | HF split for training (`flawed_fictions` or `flawed_fictions_long`) |
| `eval_source` | str | `"flawed_fictions"` | HF split for eval (`flawed_fictions` or `flawed_fictions_long`) |
| `prompt_name` | str | `"simple_boxed_prompt.txt"` | Prompt format (`simple_boxed_prompt.txt`, `conterror_detector_prompt.txt`, `conterror_detector_prompt_cot.txt`) |
| `split_ratio` | float | `0.8` | Train/eval split used only when train=eval source and counts are `-1` |
| `binary_reward_weight` | float | `1.0` | Weight for binary correctness (error/no‑error) |
| `binary_format_reward_weight` | float | `0.0` | Weight for binary format (e.g., boxed yes/no or paper tags) |
| `line_identification_reward_weight` | float | `0.0` | Weight for localization correctness (lines overlap) |
| `line_identification_format_reward_weight` | float | `0.0` | Weight for localization format (requires line tags for positives) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `binary_format` | 1 if binary is correctly formatted (boxed yes/no or paper tags), else 0 |
| `binary_accuracy` | 1 if parsed binary label equals `cont_error`, else 0 |
| `localization_format` | 1 if (binary is formatted and, for positives, both line blocks present); negatives count as 1 |
| `localization_full` | 1 if (binary is correct) and predicted error/contradicted lines overlap GT after sentence mapping, else 0 |
| `reward` | Weighted sum: `wf*binary_format + wb*binary_accuracy + wl_f*localization_format + wl*localization_full` |
