from typing import Any, Sequence

import optax

from sparse_wf.api import Schedule, TransformationArgs


def hyperbolic_schedule(init_value: float, delay: float, decay: float):
    def schedule(step):
        return init_value / (1 + step / delay) ** decay

    return schedule


def make_schedule(schedule: Schedule | str, kwargs: dict[str, dict[str, Any]]) -> optax.Schedule:
    schedule = Schedule(schedule)
    if schedule == Schedule.CONSTANT:
        return optax.constant_schedule(**kwargs["constant"])
    elif schedule == Schedule.LINEAR:
        return optax.linear_schedule(**kwargs["linear"])
    elif schedule == Schedule.EXPONENTIAL:
        return optax.exponential_decay(**kwargs["exponential"])
    elif schedule == Schedule.COSINE:
        return optax.cosine_decay_schedule(**kwargs["cosine"])
    elif schedule == Schedule.HYPERBOLIC:
        return hyperbolic_schedule(**kwargs["hyperbolic"])
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def make_optimizer(
    lr_schedule: Schedule | str,
    lr_schedule_args: dict[str, dict[str, Any]],
    transformations: Sequence[TransformationArgs],
) -> optax.GradientTransformation:
    def get_cls(name):
        if name in globals():
            return globals()[name]
        else:
            return getattr(optax, name)

    return optax.chain(
        *[get_cls(transform["name"])(*transform["args"], **transform["kwargs"]) for transform in transformations],
        optax.scale_by_schedule(make_schedule(lr_schedule, lr_schedule_args)),
        optax.scale(-1.0),
    )
