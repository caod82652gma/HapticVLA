from .tactile_sensor import TactileConfig
from .tactile_4chips import Tactile4ChipConfig, Tactile4ChipSensor
from .tactile_8chips import Tactile8ChipConfig, Tactile8ChipSensor


def make_tactile_sensor_from_config(config: TactileConfig):
    if isinstance(config, Tactile4ChipConfig):
        return Tactile4ChipSensor(config)
    if isinstance(config, Tactile8ChipConfig):
        return Tactile8ChipSensor(config)
    raise ValueError(f"Unsupported tactile config type: {type(config)}")
