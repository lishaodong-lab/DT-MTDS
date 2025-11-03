from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL
from .airl_da import DAAIRL
from .aril_sac import AIRLSAC

ALGOS = {
    'gail': GAIL,
    'airl': AIRL,
    'daairl': DAAIRL,
    'airlsac': AIRLSAC
}
