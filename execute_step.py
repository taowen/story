import cache
from get_hash_key import get_hash_key
from typing import TypedDict, Callable, Any
import pickle

if not cache.has_cache("id_map"):
    cache.update_cache(id_map={})
# key is id(xxx), value is the pickle.dumps(xxx)
id_map = cache.get_cache("id_map")

class Step(TypedDict):
    result: Any
    kwargs: dict[str, Any]

def execute_step(func: Callable, args: tuple, kwargs: dict[str, Any]) -> Step:
    """Executes a step in a pipeline, caching the result.

    Args:
        func: The function to execute.
        args: Positional arguments to pass to the function.
        kwargs: Keyword arguments to pass to the function.

    Returns:
        A dict containing the result of the function and the stringified kwargs.
    """
    key, kwargs = get_hash_key(func, args, kwargs)
    for k, v in kwargs.items():
        id_v = id(v) 
        encoded_v = pickle.dumps(v)
        if id_v in id_map and encoded_v != id_map[id_v]:
            raise ValueError(f"Value of argument {k} is mutated, please use immutable object or deepcopy before passing in.")
        id_map[id_v] = encoded_v
    if cache.has_cache(key):
        return cache.get_cache(key)
    else:
        result = func(**kwargs)
        id_result = id(result)
        encoded_result = pickle.dumps(result)
        if id_result in id_map and encoded_result != id_map[id_result]:
            raise ValueError(f"Return value of function {func.__name__} is mutated, please use immutable object or deepcopy before returning.")
        id_map[id_result] = encoded_result
        step: Step = {"result": result, "kwargs": kwargs}
        cache.update_cache(**{key: step})
        return step
