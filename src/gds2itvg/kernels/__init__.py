"""Potential kernels. All backends share the signature ``f(points (N,3), rings) -> (N,)``."""

from . import numpy_kernel

__all__ = ["get_kernel", "get_edge_kernel", "available_backends"]


def available_backends():
    names = ["numpy"]
    try:
        import numba  # noqa: F401
        names.append("numba")
    except ImportError:
        pass
    return names


def get_kernel(backend="auto"):
    """Return the polygon-potential function for ``backend`` (auto: numba, else numpy)."""
    if backend == "auto":
        try:
            from .numba_kernel import polygon_potential
            return polygon_potential
        except ImportError:
            return numpy_kernel.polygon_potential
    if backend == "numpy":
        return numpy_kernel.polygon_potential
    if backend == "numba":
        from .numba_kernel import polygon_potential
        return polygon_potential
    raise ValueError(f"unknown backend {backend!r}")


def get_edge_kernel(backend="auto"):
    """Return the weighted-edge potential function ``f(points, p1, p2, w) -> (N,)``."""
    if backend in ("auto", "numba"):
        try:
            from .numba_kernel import edge_potential
            return edge_potential
        except ImportError:
            if backend == "numba":
                raise
    if backend in ("auto", "numpy"):
        return numpy_kernel.edge_potential
    raise ValueError(f"unknown backend {backend!r}")
