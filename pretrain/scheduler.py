import math
import transformers

min_lr_ratio = 0.1


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def add_lr_decay_limit_for_cosine_schedule():
    transformers.optimization._get_cosine_schedule_with_warmup_lr_lambda = _get_cosine_schedule_with_warmup_lr_lambda
