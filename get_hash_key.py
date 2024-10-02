import hashlib
import inspect

def get_hash_key(func, args, kwargs):
    """Generates a hash key based on function source code, arguments, and keyword arguments.

    Args:
        func: The function.
        args: Positional arguments passed to the function.
        kwargs: Keyword arguments passed to the function.

    Returns:
        str: A hash key string.
    """
    source_code = inspect.getsource(func)

    # Combine args and kwargs into a single dictionary
    combined_kwargs = {**dict(zip(inspect.getfullargspec(func).args, args)), **kwargs}
    # Filter combined_kwargs to exclude arguments starting with '_'
    filtered_kwargs = {k: v for k, v in combined_kwargs.items() if not k.startswith('_')}
    
    combined_data = source_code + str(filtered_kwargs)

    return hashlib.md5(combined_data.encode()).hexdigest(), combined_kwargs
