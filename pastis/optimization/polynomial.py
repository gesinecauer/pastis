import sys
import numpy as np

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

use_jax = True
if use_jax:
    from absl import logging as absl_logging
    absl_logging.set_verbosity('error')
    from jax.config import config as jax_config
    jax_config.update("jax_platform_name", "cpu")
    jax_config.update("jax_enable_x64", True)

    import jax.numpy as ag_np
else:
    import autograd.numpy as ag_np


def _polyval(x, c, tensor=True):
    """Analagous to np.polynomial.polynomial.polyval (which is not
    differentiable with jax)"""

    c = np.array(c, ndmin=1, copy=False)
    if c.dtype.char in '?bBhHiIlLqQpP':
        # astype fails with NA
        c = c + 0.0
    if isinstance(x, (tuple, list)):
        x = ag_np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    c0 = c[-1] + x * 0
    for i in range(2, len(c) + 1):
        c0 = c[-i] + c0 * x
    return c0


def _polygrid2d(c, *args):
    """Analagous to np.polynomial.polynomial.polygrid2d (which is not
    differentiable with jax)"""

    for xi in args:
        c = _polyval(xi, c)
    return c


