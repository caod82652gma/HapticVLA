import abc
from dataclasses import dataclass

import draccus


@dataclass(kw_only=True)
class TactileConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
