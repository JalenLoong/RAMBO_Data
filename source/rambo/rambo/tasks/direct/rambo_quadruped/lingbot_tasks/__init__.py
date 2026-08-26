"""Task organization aliases; registrations remain in the simulator-native package."""

from .lift import LiftBasketQPEnvCfg
from .press import ButtonQPEnvCfg
from .pull import PullObjectQPEnvCfg
from .shoot import ShootBallQPEnvCfg

__all__ = ["ButtonQPEnvCfg", "LiftBasketQPEnvCfg", "PullObjectQPEnvCfg", "ShootBallQPEnvCfg"]
