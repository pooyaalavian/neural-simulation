from .param import ParameterSet
from .nodes import NodeParam
from .connections import ConnectionParam
from .constants import ConstantsParam
from .mixin import Mixin
from .state import State
from .model_base import ModelBase
from .plot import Plot

def deflatten(d):
    """Convert a dictionary with no nesting to a dictionary with nesting."""
    result = {}
    for key, v in d.items():
        k:list[str] = key.split('.',1)
        if len(k) == 1:
            result[k[0]] = v
        else:
            if k[0] not in result:
                result[k[0]] = {}
            result[k[0]][k[1]] = v
    return result