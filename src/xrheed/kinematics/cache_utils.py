import functools
from pathlib import Path

import dill  # type: ignore


def smart_cache(method):
    """
    Decorator for class methods that enables automatic caching of computed results.

    If the class has `use_cache = True`, the method will first attempt to load its result
    from a file named `cache_<method_name>.dill` in the directory specified by `cache_dir`.
    If the file does not exist, the method is executed and its result is saved to that file.

    Attributes expected on the class:
    - use_cache (bool): If True, enables caching behavior.
    - cache_dir (str or Path): Directory where cache files are stored. Defaults to 'cache'.


    Notes:
    - The cache filename is deterministic: `cache_<method_name>.dill`
    - Arguments to the method are not hashed or included in the filename.
    - If `use_cache` is False, the method runs normally and does not read/write cache.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "use_cache", False):
            return method(self, *args, **kwargs)

        # Build filename: cache_function_name.dill
        filename: str = f"cache_{method.__name__}.dill"
        cache_dir: Path = Path(getattr(self, "cache_dir", "cache"))

        cache_dir.mkdir(exist_ok=True)

        path = cache_dir / filename

        if path.exists():
            print(f"Loading cached result from {path}")
            with path.open("rb") as f:
                return dill.load(f)
        else:
            print(f"Cache not found. Computing and saving to {path}")
            result = method(self, *args, **kwargs)
            with path.open("wb") as f:
                dill.dump(result, f)
            return result

    return wrapper
