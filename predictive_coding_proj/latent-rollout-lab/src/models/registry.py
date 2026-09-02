from src.models.autoencoder import ae_channels
from src.models.convgru import NextFramePredictor
from src.strategies.curriculum import CurriculumStrategy
from src.strategies.scheduled_sampling import ScheduledSamplingStrategy
from src.strategies.teacher_forcing import TeacherForcingStrategy

MODELS = {
    "convgru": NextFramePredictor,
}

STRATEGIES = {
    "curriculum": CurriculumStrategy,
    "teacher_forcing": TeacherForcingStrategy,
    "scheduled_sampling": ScheduledSamplingStrategy,
}


def build_model(cfg):
    m = cfg["model"]
    name = m["name"]
    if name not in MODELS:
        raise KeyError(f"unknown model {name}; registered: {list(MODELS)}")
    return MODELS[name](
        latent_channels=ae_channels(cfg)[-1],
        hidden_channels=m["hidden_channels"],
        num_layers=m["num_layers"],
        kernel_size=m.get("kernel_size", 3),
    )


def build_strategy(cfg):
    name = cfg["strategy"]["name"]
    if name not in STRATEGIES:
        raise KeyError(f"unknown strategy {name}; registered: {list(STRATEGIES)}")
    return STRATEGIES[name](cfg)
