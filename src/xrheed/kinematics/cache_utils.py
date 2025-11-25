import functools
from pathlib import Path

import dill


def smart_cache(method):
    """
    Decorator for class methods that provides persistent, file-based caching.

    Expected instance attributes
    ----------------------------
    use_cache : bool
        Enables caching when True. If False, the method always computes
        the result without reading or writing cache files.
    cache_dir : str
        Directory where cache files are stored. Must be a string path.
        The directory will be created automatically if it does not exist.
    cache_key : str or None
        Optional suffix appended to cache filenames. Allows distinguishing
        cache files for different runs or configurations of the same method.

    Caching behavior
    ----------------
    - If `use_cache` is False:
        The method executes normally (no reads/writes).
    - If `use_cache` is True:
        The decorator builds a deterministic filename:
            cache_<method>[_<cache_key>].dill
        and stores it inside `cache_dir`.

        If the file exists, the cached result is returned.
        Otherwise, the method is executed and the result is written to disk.

    Notes
    -----
    - Method arguments are NOT included or hashed into the cache filename.
    - Cached results use `dill` for full Python object serialization.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.use_cache:
            return method(self, *args, **kwargs)

        # Use cache directory (string → Path)
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(exist_ok=True)

        # Construct filename depending on optional cache key
        if self.cache_key is None:
            filename = f"cache_{method.__name__}.dill"
        else:
            filename = f"cache_{method.__name__}_{self.cache_key}.dill"

        path = cache_dir / filename

        # Load if exists
        if path.exists():
            print(f"Loading cached result from {path}")
            with path.open("rb") as f:
                return dill.load(f)

        # Compute + save
        print(f"Cache not found. Computing and saving to {path}")
        result = method(self, *args, **kwargs)

        with path.open("wb") as f:
            dill.dump(result, f)

        return result

    return wrapper
