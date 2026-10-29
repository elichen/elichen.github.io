"""Read BAAIWorm pickles as data: every class becomes a plain stub, only numpy arrays are rebuilt."""
import pickle, sys
import numpy as np

class Stub:
    def __init__(self, *a, **k): self._args = a
    def __setstate__(self, s):
        if isinstance(s, tuple) and len(s) == 2: s = {**(s[0] or {}), **(s[1] or {})}
        self.__dict__.update(s)
    def __repr__(self): return f"<{type(self).__name__} {list(self.__dict__)[:8]}>"

ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"), ("numpy", "dtype"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
}

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            import importlib
            return getattr(importlib.import_module(module.replace("numpy.core", "numpy._core")), name)
        if module in ("builtins", "copyreg") and name in ("set", "frozenset", "complex", "slice", "_reconstructor", "object"):
            if name == "_reconstructor":
                return lambda cls, base, state: cls.__new__(cls) if isinstance(cls, type) and issubclass(cls, Stub) else (_ for _ in ()).throw(pickle.UnpicklingError("base"))
            return {"set": set, "frozenset": frozenset, "complex": complex, "slice": slice, "object": object}[name]
        return type(name, (Stub,), {"_module": module})

def load(path):
    with open(path, "rb") as f:
        return SafeUnpickler(f).load()

if __name__ == "__main__":
    o = load(sys.argv[1]); print(type(o), o if not hasattr(o, "shape") else o.shape)
