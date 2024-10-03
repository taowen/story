import hashlib
import inspect
from typing import Any, Callable, Dict, Tuple

def get_hash_key(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Generates a hash key based on function source code, arguments, and keyword arguments.

    Args:
        func: The function.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.

    Returns:
        Tuple[str, Dict[str, Any]]: A tuple containing the hash key string and 
               a dictionary of combined and filtered kwargs.
    """
    source_code = inspect.getsource(func)

    # Combine args and kwargs into a single dictionary
    combined_kwargs = {**dict(zip(inspect.getfullargspec(func).args, args)), **kwargs}
    # Filter combined_kwargs to exclude arguments starting with '_'
    filtered_kwargs = {k: v for k, v in combined_kwargs.items() if not k.startswith('_')}
    
    combined_data = source_code + str(filtered_kwargs)

    return hashlib.md5(combined_data.encode()).hexdigest(), filtered_kwargs
