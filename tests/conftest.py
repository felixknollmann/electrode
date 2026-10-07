import os

import pytest

REFERENCE = os.environ.get("GDS2ITVG_REFERENCE", "/tmp/reference")
ITVG = os.environ.get("GDS2ITVG_ITVG", "")


def backends():
    out = ["numpy"]
    try:
        import numba  # noqa: F401
        out.append("numba")
    except ImportError:
        pass
    return out


@pytest.fixture(params=backends())
def kernel(request):
    from gds2itvg.kernels import get_kernel
    return get_kernel(request.param)
