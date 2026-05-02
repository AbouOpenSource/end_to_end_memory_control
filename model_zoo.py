#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelProfile:
    name: str
    input_shape: tuple[int, ...]
    num_classes: int
    activation_units_per_sample: int
    description: str


ModelBuilder = Callable[[dict[str, Any], Any], tuple[Any, ModelProfile]]


def available_model_names() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def build_model(model_name: str, model_args: dict[str, Any], nn: Any) -> tuple[Any, ModelProfile]:
    try:
        builder = _BUILDERS[model_name]
    except KeyError as exc:
        names = ", ".join(available_model_names())
        raise ValueError(f"unknown model_name={model_name!r}; available: {names}") from exc
    return builder(dict(model_args), nn)


def make_synthetic_batch(profile: ModelProfile, batch_size: int, device: Any, torch: Any) -> tuple[Any, Any]:
    x = torch.randn((batch_size, *profile.input_shape), device=device)
    y = torch.randint(0, profile.num_classes, (batch_size,), device=device)
    return x, y


def _pop_int(args: dict[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    value = int(args.pop(key, default))
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}, got {value}")
    return value


def _reject_unknown_args(model_name: str, args: dict[str, Any]) -> None:
    if args:
        unknown = ", ".join(sorted(args))
        raise ValueError(f"unknown model_args for {model_name}: {unknown}")


def _build_linear(args: dict[str, Any], nn: Any) -> tuple[Any, ModelProfile]:
    input_dim = _pop_int(args, "input_dim", 256)
    num_classes = _pop_int(args, "num_classes", 10)
    _reject_unknown_args("linear", args)

    model = nn.Linear(input_dim, num_classes)
    profile = ModelProfile(
        name="linear",
        input_shape=(input_dim,),
        num_classes=num_classes,
        activation_units_per_sample=input_dim + num_classes,
        description="single linear classifier",
    )
    return model, profile


def _build_mlp(args: dict[str, Any], nn: Any) -> tuple[Any, ModelProfile]:
    input_dim = _pop_int(args, "input_dim", 256)
    hidden_dim = _pop_int(args, "hidden_dim", 512)
    hidden_layers = _pop_int(args, "hidden_layers", 2)
    num_classes = _pop_int(args, "num_classes", 10)
    _reject_unknown_args("mlp", args)

    layers: list[Any] = []
    in_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_classes))

    model = nn.Sequential(*layers)
    profile = ModelProfile(
        name="mlp",
        input_shape=(input_dim,),
        num_classes=num_classes,
        activation_units_per_sample=input_dim + hidden_layers * hidden_dim + num_classes,
        description=f"{hidden_layers}-hidden-layer MLP",
    )
    return model, profile


def _build_tiny_cnn(args: dict[str, Any], nn: Any) -> tuple[Any, ModelProfile]:
    channels = _pop_int(args, "channels", 3)
    image_size = _pop_int(args, "image_size", 32)
    width = _pop_int(args, "width", 16)
    num_classes = _pop_int(args, "num_classes", 10)
    _reject_unknown_args("tiny_cnn", args)

    model = nn.Sequential(
        nn.Conv2d(channels, width, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(width * 4, num_classes),
    )

    h2 = math.ceil(image_size / 2)
    h4 = math.ceil(h2 / 2)
    activation_units = (
        channels * image_size * image_size
        + width * image_size * image_size
        + (width * 2) * h2 * h2
        + (width * 4) * h4 * h4
        + num_classes
    )
    profile = ModelProfile(
        name="tiny_cnn",
        input_shape=(channels, image_size, image_size),
        num_classes=num_classes,
        activation_units_per_sample=activation_units,
        description="small convolutional image classifier",
    )
    return model, profile


_BUILDERS: dict[str, ModelBuilder] = {
    "linear": _build_linear,
    "mlp": _build_mlp,
    "tiny_cnn": _build_tiny_cnn,
}
