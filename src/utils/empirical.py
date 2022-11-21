# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute empirical NNGP and NTK; approximate functions via Taylor series.

All functions in this module are applicable to any JAX functions of proper
signatures (not only those from `nt.stax`).

NNGP and NTK are computed using `empirical_nngp_fn`, `empirical_ntk_fn`, or
`empirical_kernel_fn` (for both). The kernels have a very specific output shape
convention that may be unexpected. Further, NTK has multiple implementations
that may perform differently depending on the task. Please read individual
functions' docstrings.

Example:
  >>>  from jax import random
  >>>  import neural_tangents as nt
  >>>  from neural_tangents import stax
  >>>
  >>>  key1, key2, key3 = random.split(random.PRNGKey(1), 3)
  >>>  x_train = random.normal(key1, (20, 32, 32, 3))
  >>>  y_train = random.uniform(key1, (20, 10))
  >>>  x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>>  # A narrow CNN.
  >>>  init_fn, f, _ = stax.serial(
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu(),
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Flatten(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Default setting: reducing over logits; pass `vmap_axes=0` because the
  >>>  # network is iid along the batch axis, no BatchNorm. Use default
  >>>  # `implementation=1` since the network has few trainable parameters.
  >>>  kernel_fn = nt.empirical_kernel_fn(f, trace_axes=(-1,),
  >>>                                     vmap_axes=0, implementation=1)
  >>>
  >>>  # (5, 20) np.ndarray test-train NNGP/NTK
  >>>  nngp_test_train = kernel_fn(x_test, x_train, 'nngp', params)
  >>>  ntk_test_train = kernel_fn(x_test, x_train, 'ntk', params)
  >>>
  >>>  # Full kernel: not reducing over logits.
  >>>  kernel_fn = nt.empirical_kernel_fn(f, trace_axes=(), vmap_axes=0)
  >>>
  >>>  # (5, 20, 10, 10) np.ndarray test-train NNGP/NTK namedtuple.
  >>>  k_test_train = kernel_fn(x_test, x_train, params)
  >>>
  >>>  # A wide FCN with lots of parameters
  >>>  init_fn, f, _ = stax.serial(
  >>>      stax.Flatten(),
  >>>      stax.Dense(1024),
  >>>      stax.Relu(),
  >>>      stax.Dense(1024),
  >>>      stax.Relu(),
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Use implicit differentiation in NTK: `implementation=2` to reduce
  >>>  # memory cost, since the network has many trainable parameters.
  >>>  ntk_fn = nt.empirical_ntk_fn(f, vmap_axes=0, implementation=2)
  >>>
  >>>  # (5, 5) np.ndarray test-test NTK
  >>>  ntk_test_train = ntk_fn(x_test, None, params)
  >>>
  >>>  # Compute only output variances:
  >>>  nngp_fn = nt.empirical_nngp_fn(f, diagonal_axes=(0,))
  >>>
  >>>  # (20,) np.ndarray train-train diagonal NNGP
  >>>  nngp_train_train_diag = nngp_fn(x_train, None, params)
"""
"""Fast computation of empirical NTK.

All functions in this module are applicable to any JAX functions of proper
signatures.

The NTK kernels have a very specific output shape convention that may be
unexpected. Further, NTK has multiple implementations that may perform
differently depending on the task.
Please read individual functions' docstrings.

Example:
  >>>  from jax import random
  >>>  from fast_finite_width_ntk import empirical_ntk_fn
  >>>  from jax.experimental import stax
  >>>
  >>>  key1, key2, key3 = random.split(random.PRNGKey(1), 3)
  >>>  x_train = random.normal(key1, (20, 32, 32, 3))
  >>>  y_train = random.uniform(key1, (20, 10))
  >>>  x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>>  # A CNN.
  >>>  init_fn, f = stax.serial(
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu,
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Relu,
  >>>      stax.Conv(32, (3, 3)),
  >>>      stax.Flatten,
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Default setting: reducing over logits; pass `vmap_axes=0` because the
  >>>  # network is iid along the batch axis, no BatchNorm. Using
  >>>  # structured derivatives since forward pass is expensive relative to
  >>>  # the size of parameters.
  >>>  kernel_fn = empirical_ntk_fn(f, trace_axes=(-1,), vmap_axes=0)
  >>>
  >>>  # (5, 20) np.ndarray test-train NTK
  >>>  nngp_test_train = kernel_fn(x_test, x_train, params)
  >>>  ntk_test_train = kernel_fn(x_test, x_train, params)
  >>>
  >>>  # Full kernel: not reducing over logits.
  >>>  kernel_fn = empirical_ntk_fn(f, trace_axes=(), vmap_axes=0)
  >>>
  >>>  # (5, 20, 10, 10) np.ndarray test-train NTK.
  >>>  k_test_train = kernel_fn(x_test, x_train, params)
  >>>
  >>>  # An FCN
  >>>  init_fn, f = stax.serial(
  >>>      stax.Flatten,
  >>>      stax.Dense(1024),
  >>>      stax.Relu,
  >>>      stax.Dense(1024),
  >>>      stax.Relu,
  >>>      stax.Dense(10)
  >>>  )
  >>>
  >>>  _, params = init_fn(key3, x_train.shape)
  >>>
  >>>  # Use ntk-vector products since the network has many parameters
  >>>  # relative to the cost of forward pass.
  >>>  ntk_fn = empirical_ntk_fn(f, vmap_axes=0, implementation=2)
  >>>
  >>>  # (5, 5) np.ndarray test-test NTK
  >>>  ntk_test_test = ntk_fn(x_test, None, params)
  >>>
  >>>  # Compute only NTK diagonal variances:
  >>>  ntk_fn = empirical_ntk_fn(f, diagonal_axes=(0,))
  >>>
  >>>  # (20,) np.ndarray train-train NTK diagonal
  >>>  ntk_train_train_diag = ntk_fn(x_train, None, params)
"""
import functools
import enum
import operator
from typing import Sequence, Union, Callable, Optional, Tuple, Dict, Any, List, Set
from jax import eval_shape, jacobian, jvp, vjp, vmap
from jax._src.api import linear_transpose, _check_callable
import jax.numpy as np
import jax
import warnings
from jax import core, lax
from jax.interpreters import ad, xla
from jax import linear_util as lu
import numpy as onp
from jax.core import Jaxpr, JaxprEqn, Var, Literal, ShapedArray
from jax.interpreters.ad import UndefinedPrimal, Zero
from jax._src.util import safe_map, safe_zip, partial
from jax.tree_util import tree_transpose, tree_structure, \
  Partial, tree_flatten, tree_unflatten, tree_reduce, tree_map
from src.utils import neural_utils as utils
from src.utils import neural_rules as rules
from neural_tangents.utils.typing import ApplyFn, NTTree, PyTree, Axes, VMapAxes


zip = safe_zip
map = safe_map


class NtkImplementation(enum.IntEnum):
  """Implementation method of the underlying computation."""
  AUTO = 0
  JACOBIAN_CONTRACTION = 1
  NTK_VECTOR_PRODUCTS = 2
  STRUCTURED_DERIVATIVES = 3


def empirical_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes = (-1,),
    diagonal_axes: Axes = (),
    vmap_axes: VMapAxes = None,
    implementation: Union[NtkImplementation, int] = NtkImplementation.STRUCTURED_DERIVATIVES,
    fwd: Optional[bool] = None,
    j_rules: bool = True,
    a_rules: bool = True,
) -> Callable[[NTTree[np.ndarray],
               Optional[NTTree[np.ndarray]],
               PyTree],
              NTTree[np.ndarray]]:
  r"""Returns a function to draw a single sample the NTK of a given network `f`.

  The Neural Tangent Kernel is defined as :math:`J(X_1) J(X_2)^T` where
  :math:`J` is the Jacobian :math:`df/dparams` of shape
  `full_output_shape + params.shape`.

  For best performance:
  1) pass `x2=None` if `x1 == x2;
  2) prefer square batches (i.e `x1.shape == x2.shape`);
  3) make sure to set `vmap_axes` correctly.
  4) try different `implementation` values.

  WARNING: Resulting kernel shape is *nearly* `zip(f(x1).shape, f(x2).shape)`
  subject to `trace_axes` and `diagonal_axes` parameters, which make certain
  assumptions about the outputs `f(x)` that may only be true in the infinite
  width / infinite number of samples limit, or may not apply to your
  architecture. For most precise results in the context of linearized training
  dynamics of a specific finite-width network, set both `trace_axes=()` and
  `diagonal_axes=()` to obtain the kernel exactly of shape
  `zip(f(x1).shape, f(x2).shape)`.

  For networks with multiple (i.e. lists, tuples, PyTrees) outputs, in principal
  the empirical kernels will have terms measuring the covariance between the
  outputs. Here, we ignore these cross-terms and consider each output
  separately. Please raise an issue if this feature is important to you.

  Args:
    f:
      the function whose NTK we are computing. `f` should have the signature
      `f(params, inputs[, rng])` and should return an `np.ndarray` outputs.

    trace_axes:
      output axes to trace the output kernel over, i.e. compute only the trace
      of the covariance along the respective pair of axes (one pair for each
      axis in `trace_axes`). This allows to save space and compute if you are
      only interested in the respective trace, but also improve approximation
      accuracy if you know that covariance along these pairs of axes converges
      to a `constant * identity matrix` in the limit of interest (e.g.
      infinite width or infinite `n_samples`). A common use case is the channel
      / feature / logit axis, since activation slices along such axis are i.i.d.
      and the respective covariance along the respective pair of axes indeed
      converges to a constant-diagonal matrix in the infinite width or infinite
      `n_samples` limit.
      Also related to "contracting dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    diagonal_axes:
      output axes to diagonalize the output kernel over, i.e. compute only the
      diagonal of the covariance along the respective pair of axes (one pair for
      each axis in `diagonal_axes`). This allows to save space and compute, if
      off-diagonal values along these axes are not needed, but also improve
      approximation accuracy if their limiting value is known theoretically,
      e.g. if they vanish in the limit of interest (e.g. infinite
      width or infinite `n_samples`). If you further know that on-diagonal
      values converge to the same constant in your limit of interest, you should
      specify these axes in `trace_axes` instead, to save even more compute and
      gain even more accuracy. A common use case is computing the variance
      (instead of covariance) along certain axes.
      Also related to "batch dimensions" in XLA terms.
      (https://www.tensorflow.org/xla/operation_semantics#dotgeneral)

    vmap_axes:
      A triple of `(in_axes, out_axes, kwargs_axes)`
      passed to `vmap` to evaluate the empirical NTK in parallel ove these axes.
      Precisely, providing this argument implies that `f(params, x, **kwargs)`
      equals to a concatenation along `out_axes` of `f` applied to slices of
      `x` and `**kwargs` along `in_axes` and `kwargs_axes`. In other words, it
      certifies that `f` can be evaluated as a `vmap` with `out_axes=out_axes`
      over `x` (along `in_axes`) and those arguments in `**kwargs` that are
      present in `kwargs_axes.keys()` (along `kwargs_axes.values()`).

      For example if `_, f, _ = nt.stax.Aggregate()`, `f` is called via
      `f(params, x, pattern=pattern)`. By default, inputs `x`, patterns
      `pattern`, and outputs of `f` are all batched along the leading `0`
      dimension, and each output `f(params, x, pattern=pattern)[i]` only
      depends on the inputs `x[i]` and `pattern[i]`. In this case, we can
      pass `vmap_axes=(0, 0, dict(pattern=0)` to specify along which dimensions
      inputs, outputs, and keyword arguments are batched respectively.

      This allows us to evaluate Jacobians much more
      efficiently. If `vmap_axes` is not a triple, it is interpreted as
      `in_axes = out_axes = vmap_axes, kwargs_axes = {}`. For example a very
      common use case is `vmap_axes=0` for a neural network with leading (`0`)
      batch dimension, both for inputs and outputs, and no interactions between
      different elements of the batch (e.g. no BatchNorm, and, in the case of
      `nt.stax`, also no Dropout). However, if there is interaction between
      batch elements or no concept of a batch axis at all, `vmap_axes` must be
      set to `None`, to avoid wrong (and potentially silent) results.

    implementation:
      `1`, `2`, `3`, or `0`.

      `0` selects the best of `1`, `2`, and `3` based on FLOPs analysis.
      It only works correctly for TPUs, and on CPU/GPU returns wrong FLOPs and
      may select a slower method.

      `1` directly instantiates Jacobians and computes their contraction.

      `2` uses NTK-vector products to avoid expensive contraction at the
      cost of extra forward and backward passes through the network.

      `3` uses structured derivatives to simplify the NTK contraction.

    j_rules:
      `True` to allow custom Jacobian rules for `dy/dw` computations.

    a_rules:
      `True` to allow simplification rules for structured `dy/dw` derivatives.

    fwd:
      `True` to allow `jvp` in intermediary kernel computations, `False` to
      always use `vjp`. `None` to decide based on input/output sizes.

  Returns:
    A function `ntk_fn` that computes the empirical ntk.
  """
  return {
      NtkImplementation.JACOBIAN_CONTRACTION: _jacobian_contraction_ntk_fn,
      NtkImplementation.NTK_VECTOR_PRODUCTS: _ntk_vector_products_ntk_fn,
      NtkImplementation.STRUCTURED_DERIVATIVES: _structured_derivatives_ntk_fn,
      NtkImplementation.AUTO: _empirical_auto_ntk_fn,
  }[implementation](f=f,
                    trace_axes=trace_axes,
                    diagonal_axes=diagonal_axes,
                    vmap_axes=vmap_axes,
                    fwd=fwd,
                    j_rules=j_rules,
                    a_rules=a_rules)


def _ntk_vector_products_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> Callable[[NTTree[np.ndarray],
               Optional[NTTree[np.ndarray]],
               PyTree],
              NTTree[np.ndarray]]:
  """Compute NTK via NTK-vector products."""

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the empirical NTK with NTK-vector products.

    Args:
      x1:
        first batch of inputs.

      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.

      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.

      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1 != x2`)
        or same (if `x1 == x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def get_ntk(x1, x2, *args):
      f1, f2 = _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args, x1, x2)

      def delta_vjp_jvp(delta):
        def delta_vjp(delta):
          return vjp(f2, params)[1](delta)
        return jvp(f1, (params,), delta_vjp(delta))[1]

      fx1, fx2 = eval_shape(f1, params), eval_shape(f2, params)
      eye = utils.std_basis(fx1)
      ntk = vmap(linear_transpose(delta_vjp_jvp, fx2))(eye)
      ntk = tree_map(lambda fx12: utils.unravel_array_into_pytree(fx1, 0, fx12),
                     ntk)
      ntk = _diagonal(ntk, fx1)
      return ntk

    if x_axis is not None or kw_axes:
      x2 = x1 if utils.all_none(x2) else x2

      kw_in_axes = [kw_axes[k] if k in kw_axes else None for k in keys]
      in_axes1 = [x_axis, None] + kw_in_axes + [None] * len(kw_in_axes)
      in_axes2 = [None, x_axis] + [None] * len(kw_in_axes) + kw_in_axes

      get_ntk = vmap(vmap(get_ntk,
                          in_axes1,
                          fx_axis),
                     in_axes2,
                     _add(fx_axis, _ndim(fx1)))

    return _trace_and_diagonal(get_ntk(x1, x2, *args1, *args2),
                               trace_axes, diagonal_axes)

  return ntk_fn


def _jacobian_contraction_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    **kwargs
) -> Callable[[NTTree[np.ndarray],
               Optional[NTTree[np.ndarray]],
               PyTree],
              NTTree[np.ndarray]]:
  """Compute NTK by directly instantiating Jacobians and contracting."""

  @utils.nt_tree_fn(tree_structure_argnum=0)
  def sum_and_contract(fx, j1, j2):
    ndim = fx.ndim
    size = utils.size_at(fx, trace_axes)

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

    def contract(x, y):
      param_axes = list(range(x.ndim))[ndim:]
      contract_axes = _trace_axes + param_axes
      return utils.dot_general(x, y, contract_axes, _diagonal_axes) / size

    return tree_reduce(operator.add, tree_map(contract, j1, j2))

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the empirical NTK (jacobian outer product).

    Args:
      x1:
        first batch of inputs.

      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.

      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.

      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
        f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def j_fn(x, *args):
      _kwargs = {k: v for k, v in zip(keys, args)}
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
      jx = jacobian(fx)(params)
      return jx

    if x_axis is not None or kw_axes:
      in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

    j1 = j_fn(x1, *args1)
    j2 = j_fn(x2, *args2) if not utils.all_none(x2) else j1
    ntk = sum_and_contract(fx1, j1, j2)
    return ntk

  return ntk_fn


def _structured_derivatives_ntk_fn(
    f: ApplyFn,
    trace_axes: Axes,
    diagonal_axes: Axes,
    vmap_axes: VMapAxes,
    fwd: Optional[bool],
    j_rules: bool,
    a_rules: bool,
) -> Callable[[NTTree[np.ndarray],
               Optional[NTTree[np.ndarray]],
               PyTree],
              NTTree[np.ndarray]]:
  """Compute NTK by using structured derivatives."""

  @utils.nt_tree_fn(tree_structure_argnum=0, nargs=5)
  def sum_and_contract(fx1,
                       fx2,
                       fx_axis,
                       df_dys_1,
                       df_dys_2,
                       dy_dws_1,
                       dy_dws_2,
                       dtype
  ):
    ndim = fx1.ndim
    size = utils.size_at(fx1, trace_axes)

    _diagonal_axes = utils.canonicalize_axis(diagonal_axes, ndim)
    _trace_axes = utils.canonicalize_axis(trace_axes, ndim)

    def contract(df_dys_1, df_dys_2, dy_dws_1, dy_dws_2):
      ntk = np.zeros((), dtype=dtype)

      for i_1, (df_dy_1, dy_dw_1_) in enumerate(zip(df_dys_1, dy_dws_1)):
        for i_2, (df_dy_2, dy_dw_2_) in enumerate(zip(df_dys_2, dy_dws_2)):
          dy_dw_1, axes_1 = dy_dw_1_
          dy_dw_2, axes_2 = dy_dw_2_

          df_dy_dims_1, df_dy_dims_2, out_dims = _get_dims(df_dy_1,
                                                           df_dy_2,
                                                           ndim,
                                                           _trace_axes,
                                                           _diagonal_axes)

          if len(axes_1.out_trace) == len(axes_2.out_trace):
            for i, (id_1, id_2) in enumerate(zip(axes_1.out_trace,
                                                 axes_2.out_trace)):
              axis_id = df_dy_1.ndim + df_dy_2.ndim + i
              y_axis_1 = id_1 % (df_dy_1.ndim - ndim)
              y_axis_2 = id_2 % (df_dy_2.ndim - ndim)
              df_dy_dims_1[ndim + y_axis_1] = axis_id
              df_dy_dims_2[ndim + y_axis_2] = axis_id
          else:
            raise NotImplementedError('Different number of trace_axes 1/2.')

          dy_dw_dims_1 = list(range(-dy_dw_1.ndim, 0))
          dy_dw_dims_2 = list(range(-dy_dw_2.ndim, 0))

          if fx_axis is not None:
            df_dy_1 = np.moveaxis(df_dy_1, 0, fx_axis)
            df_dy_2 = np.moveaxis(df_dy_2, 0, fx_axis)

            dy_dw_dims_1[0] = df_dy_dims_1[fx_axis]
            dy_dw_dims_2[0] = df_dy_dims_2[fx_axis]
            ix_1, ix_2 = 1, 1

          else:
            ix_1, ix_2 = 0, 0

          if len(axes_1.out_diagonal) == len(axes_2.out_diagonal):
            for i, (id_1, id_2) in enumerate(zip(axes_1.out_diagonal,
                                                 axes_2.out_diagonal)):
              axis_id = (-100 -df_dy_1.ndim - df_dy_2.ndim - dy_dw_1.ndim
                         - dy_dw_2.ndim - i)

              df_dy_dims_1[ndim + id_1] = axis_id
              dy_dw_dims_1[ix_1 + id_1] = axis_id

              df_dy_dims_2[ndim + id_2] = axis_id
              dy_dw_dims_2[ix_2 + id_2] = axis_id
          else:
            raise NotImplementedError('Different number of diagonal_axes 1/2.')

          for i in range(ndim, df_dy_1.ndim):
            if i - ndim not in (axes_1.out_trace +
                                axes_1.out_diagonal +
                                axes_1.out_broadcast):
              dy_dw_dims_1[ix_1] = df_dy_dims_1[i]
            ix_1 += 1

          for i in range(ndim, df_dy_2.ndim):
            if i - ndim not in (axes_2.out_trace +
                                axes_2.out_diagonal +
                                axes_2.out_broadcast):
              dy_dw_dims_2[ix_2] = df_dy_dims_2[i]
            ix_2 += 1

          def check_dims(arrays, dims):
            for idx_1, (a1, dims_1) in enumerate(zip(arrays, dims)):
              if len(set(dims_1)) != len(dims_1):
                raise ValueError(f'Dimensions {idx_1} contain duplicate axes: '
                                 f'{dims_1}.')

              for ax_1, dim_1 in enumerate(dims_1):
                sz_idx_1 = a1.shape[ax_1]
                for idx_2, (a2, dims_2) in enumerate(zip(arrays, dims)):
                  if dim_1 in dims_2:
                    ax_2 = dims_2.index(dim_1)
                    sz_idx_2 = a2.shape[ax_2]
                    if sz_idx_2 != sz_idx_1:
                      raise ValueError(f'Arrays {idx_1} and {idx_2} mismatch '
                                       f'sizes at {ax_1} and {ax_2}: '
                                       f'{sz_idx_1} != {sz_idx_2}')

          check_dims(
              arrays=[df_dy_1, dy_dw_1, dy_dw_2, df_dy_2],
              dims=[df_dy_dims_1, dy_dw_dims_1, dy_dw_dims_2, df_dy_dims_2]
          )

          ntk_l = np.einsum(
              df_dy_1, df_dy_dims_1,
              dy_dw_1, dy_dw_dims_1,
              dy_dw_2, dy_dw_dims_2,
              df_dy_2, df_dy_dims_2,
              out_dims
          )
          ntk += ntk_l

      return ntk

    ntk = tree_reduce(
        operator.add,
        tree_map(
            contract,
            df_dys_1, df_dys_2, dy_dws_1, dy_dws_2,
            is_leaf=lambda x: x == [] or isinstance(x, list) and isinstance(x[0], np.ndarray)),
        np.zeros((), dtype)
    )
    ntk /= size
    ntk_shape = _ntk_shape(fx1.shape, fx2.shape, trace_axes, diagonal_axes)
    ntk = np.broadcast_to(ntk, ntk_shape)  # if ntk is 0.
    return ntk

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the structured derivatives NTK.

    Args:
      x1:
        first batch of inputs.

      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.

      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.

      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """
    args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis = _get_args(
      f, apply_fn_kwargs, params, vmap_axes, x1, x2)

    def j_fn(x, *args):
      _kwargs = {k: v for k, v in zip(keys, args)}
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes, **_kwargs)
      jx = _get_df_dys_and_dy_dws(fx,
                                  fwd=fwd,
                                  j_rules=j_rules,
                                  a_rules=a_rules)(params)
      return jx

    if x_axis is not None or kw_axes:
      in_axes = [x_axis] + [kw_axes[k] if k in kw_axes else None for k in keys]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=0)

    df_dys_1, dy_dws_1 = j_fn(x1, *args1)
    df_dys_2, dy_dws_2 = j_fn(x2, *args2) if not utils.all_none(x2) else (
      df_dys_1, dy_dws_1)

    fx_axis, dtype = _get_fx_axis_and_dtype(fx1, fx_axis, params)
    ntk = sum_and_contract(fx1,
                           fx2,
                           fx_axis,
                           df_dys_1,
                           df_dys_2,
                           dy_dws_1,
                           dy_dws_2,
                           dtype)

    return ntk

  return ntk_fn


def _empirical_auto_ntk_fn(**kwargs
) -> Callable[[NTTree[np.ndarray],
               Optional[NTTree[np.ndarray]],
               PyTree],
              NTTree[np.ndarray]]:
  """Compute NTK by automatically selecting the best implementation.

  Returns wrong FLOPS on CPU and GPU when JITting.
  """

  cache = {}

  def ntk_fn(x1: NTTree[np.ndarray],
             x2: Optional[NTTree[np.ndarray]],
             params: PyTree,
             **apply_fn_kwargs) -> np.ndarray:
    """Computes a single sample of the automatic empirical NTK.

    Args:
      x1:
        first batch of inputs.
      x2:
        second batch of inputs. `x2=None` means `x2=x1`. `f(x2)` must have a
        matching shape with `f(x1)` on `trace_axes` and `diagonal_axes`.
      params:
        A `PyTree` of parameters about which we would like to compute the
        neural tangent kernel.
      **apply_fn_kwargs:
        keyword arguments passed to `apply_fn`. `apply_fn_kwargs` will be split
        into `apply_fn_kwargs1` and `apply_fn_kwargs2` by the `split_kwargs`
        function which will be passed to `apply_fn`. In particular, the rng key
        in `apply_fn_kwargs`, will be split into two different (if `x1!=x2`) or
        same (if `x1==x2`) rng keys. See the `_read_key` function for more
        details.

    Returns:
      A single sample of the empirical NTK. The shape of the kernel is "almost"
      `zip(f(x1).shape, f(x2).shape)` except for:
      1) `trace_axes` are absent as they are contracted over.
      2) `diagonal_axes` are present only once.
      All other axes are present twice.
    """
    shapes = tree_map(np.shape, (x1, x2, params, apply_fn_kwargs))
    shapes = _to_tuple_tree(shapes)

    if shapes not in cache:
      best_ntk_fn = None
      best_flops = onp.inf
      for implementation in NtkImplementation:
        if implementation != NtkImplementation.AUTO:
          ntk_fn = empirical_ntk_fn(**kwargs, implementation=implementation)
          flops = utils.get_flops(ntk_fn, True, x1, x2, params,
                                  **apply_fn_kwargs)
          print(f'impl={implementation}, flops={flops}')
          if flops < best_flops:
            best_flops = flops
            best_ntk_fn = ntk_fn

      if best_ntk_fn is None:
        raise ValueError('This should not happen.')
      cache[shapes] = best_ntk_fn

    return cache[shapes](x1, x2, params, **apply_fn_kwargs)

  return ntk_fn


# INTERNAL UTILITIES


@utils.nt_tree_fn(nargs=1)
def _trace_and_diagonal(ntk: np.ndarray,
                        trace_axes: Axes,
                        diagonal_axes: Axes) -> np.ndarray:
  """Extract traces and diagonals along respective pairs of axes from the `ntk`.

  Args:
    ntk:
      input empirical NTK of shape `(N1, X, Y, Z, ..., N2, X, Y, Z, ...)`.
    trace_axes:
      axes (among `X, Y, Z, ...`) to trace over, i.e. compute the trace along
      and remove the  respective pairs of axes from the `ntk`.
    diagonal_axes:
      axes (among `X, Y, Z, ...`) to take the diagonal along, i.e. extract the
      diagonal along the respective pairs of axes from the `ntk` (and hence
      reduce the resulting `ntk` axes count by 2).
  Returns:
    An array of shape, for example, `(N1, N2, Y, Z, Z, ...)` if
    `trace_axes=(1,)` (`X` axes removed), and `diagonal_axes=(2,)` (`Y` axes
    replaced with a single `Y` axis).
  """

  if ntk.ndim % 2 == 1:
    raise ValueError('Expected an even-dimensional kernel.')

  output_ndim = ntk.ndim // 2

  trace_axes = utils.canonicalize_axis(trace_axes, output_ndim)
  diagonal_axes = utils.canonicalize_axis(diagonal_axes, output_ndim)

  n_diag, n_trace = len(diagonal_axes), len(trace_axes)
  contract_size = utils.size_at(ntk.shape[:output_ndim], trace_axes)

  for i, c in enumerate(reversed(trace_axes)):
    ntk = np.trace(ntk, axis1=c, axis2=output_ndim + c - i)

  for i, d in enumerate(diagonal_axes):
    axis1 = d - i
    axis2 = output_ndim + d - 2 * i - n_trace
    for c in trace_axes:
      if c < d:
        axis1 -= 1
        axis2 -= 1
    ntk = np.diagonal(ntk, axis1=axis1, axis2=axis2)

  ntk = utils.zip_axes(ntk, 0, ntk.ndim - n_diag)
  res_diagonal_axes = utils.get_res_batch_dims(trace_axes, diagonal_axes)
  ntk = np.moveaxis(ntk, range(-n_diag, 0), res_diagonal_axes)
  return ntk / contract_size


def _get_f_params(f, x, x_axis, fx_axis, kw_axes, **apply_fn_kwargs):
  x = _expand_dims(x, x_axis)

  apply_fn_kwargs = {
      k: _expand_dims(v, kw_axes[k]) if k in kw_axes else v
      for k, v in apply_fn_kwargs.items()
  }

  def _f(p):
    fx = f(p, x, **apply_fn_kwargs)
    fx = utils.get_masked_array(fx)

    get_masked = utils.nt_tree_fn()(lambda o: o.masked_value)
    fx = get_masked(fx)
    return _squeeze(fx, fx_axis)

  return _f


def _get_args(f, apply_fn_kwargs, params, vmap_axes, x1, x2):
  kwargs1, kwargs2 = utils.split_kwargs(apply_fn_kwargs, x1, x2)

  @utils.nt_tree_fn()
  def unmask(x):
    return x.masked_value if isinstance(x, utils.MaskedArray) else x

  fx1 = unmask(eval_shape(f, params, x1, **kwargs1))
  fx2 = fx1 if utils.all_none(x2) else unmask(eval_shape(f, params, x2, **kwargs2))

  x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x1, fx1, **kwargs1)

  keys = apply_fn_kwargs.keys()
  args1 = tuple(kwargs1[k] for k in keys)
  args2 = tuple(kwargs2[k] for k in keys)
  return args1, args2, fx1, fx2, fx_axis, keys, kw_axes, x_axis


def _get_f1_f2(f, keys, x_axis, fx_axis, kw_axes, args, x1, x2):
  args1, args2 = args[:len(args) // 2], args[len(args) // 2:]
  _kwargs1 = {k: v for k, v in zip(keys, args1)}
  _kwargs2 = {k: v for k, v in zip(keys, args2)}
  f1 = _get_f_params(f, x1, x_axis, fx_axis, kw_axes, **_kwargs1)
  f2 = f1 if utils.all_none(x2) else _get_f_params(
      f, x2, x_axis, fx_axis, kw_axes, **_kwargs2)
  return f1, f2


def _expand_dims_array(x: np.ndarray, axis: int):
  if not isinstance(axis, int):
    raise TypeError(axis, type(axis))

  if isinstance(x, (Zero, UndefinedPrimal)):
    raise TypeError(x)

  def expand(x):
    return np.expand_dims(x, axis)

  if isinstance(x, ShapedArray):
    return eval_shape(expand, x)

  elif isinstance(x, np.ndarray):
    return expand(x)

  else:
    raise ValueError(type(x), x)


def _expand_dims(x, axis):
  if axis is None or x is None:
    return x
  return tree_map(_expand_dims_array, x, axis)


def _add(x, y):
  if x is None or y is None:
    return None
  return tree_map(operator.add, x, y)


def _sub(x, y):
  return tree_map(operator.sub, x, y)


def _div(x, y):
  return tree_map(lambda x: x / y, x)


def _squeeze(x, axis):
  if axis is None:
    return x

  return tree_map(utils.squeeze, x, axis)


@utils.nt_tree_fn()
def _ndim(x):
  return x.ndim


def _mod(x, y):
  return tree_map(operator.mod, x, y)


def _unmask(x):
  masked_output = utils.get_masked_array(x)
  return utils.nt_tree_fn()(lambda x: x.masked_value)(masked_output)



def _diagonal(ntk, fx):
  ntk_flat, _ = tree_flatten(ntk)
  fx_flat, fx_tree = tree_flatten(fx)
  n = len(fx_flat)
  diag = [ntk_flat[i * (n + 1)] for i in range(n)]
  return tree_unflatten(fx_tree, diag)


def _canonicalize_axes(vmap_axes: Optional[VMapAxes],
                       x: NTTree[np.ndarray],
                       fx: NTTree[np.ndarray],
                       **kwargs) -> VMapAxes:
  if isinstance(vmap_axes, tuple) and len(vmap_axes) == 3:
    x_axis, fx_axis, kw_axes = vmap_axes
  else:
    x_axis, fx_axis, kw_axes = vmap_axes, vmap_axes, {}

  is_leaf = lambda x: isinstance(x, (np.ndarray, utils.MaskedArray))

  if isinstance(x_axis, int):
    x_axis = tree_map(lambda _: x_axis, x, is_leaf=is_leaf)

  if isinstance(fx_axis, int):
    fx_axis = tree_map(lambda _: fx_axis, fx, is_leaf=is_leaf)

  if isinstance(kw_axes, int):
    kw_axes = tree_map(lambda _: kw_axes, kwargs, is_leaf=is_leaf)

  x_axis = _mod(x_axis, _ndim(x)) if x_axis is not None else None
  fx_axis = _mod(fx_axis, _ndim(fx)) if fx_axis is not None else None
  kw_axes = _mod(kw_axes, {k: _ndim(kwargs[k]) for k in kw_axes})
  return x_axis, fx_axis, kw_axes


def _to_tuple_tree(x):
  """Replace all lists and dictionaries with tuples in a PyTree for hashing."""
  if isinstance(x, (tuple, list)):
    return tuple(_to_tuple_tree(x_i) for x_i in x)

  if isinstance(x, dict):
    return tuple((k, v) for k, v in sorted(x.items()))

  return x


def _ntk_shape(fx1_shape, fx2_shape, trace_axes, diagonal_axes):
  ntk_shape = ()

  trace_axes = utils.canonicalize_axis(trace_axes, fx1_shape)
  diagonal_axes = utils.canonicalize_axis(diagonal_axes, fx1_shape)

  for i, (a1, a2) in enumerate(zip(fx1_shape, fx2_shape)):
    if i not in trace_axes:
      if i in diagonal_axes:
        assert a1 == a2
        ntk_shape += (a1,)
      else:
        ntk_shape += (a1, a2)
    else:
      assert a1 == a2
  return ntk_shape


def _get_dims(df_dy_1, df_dy_2, ndim, trace_axes, diagonal_axes):
  df_dy_dims_1 = list(range(df_dy_1.ndim))
  df_dy_dims_2 = list(range(df_dy_1.ndim, df_dy_1.ndim + df_dy_2.ndim))

  out_dims = []
  for i in range(ndim):
    if i in trace_axes:
      assert df_dy_1.shape[i] == df_dy_2.shape[i]
      df_dy_dims_2[i] = df_dy_dims_1[i]
    elif i in diagonal_axes:
      assert df_dy_1.shape[i] == df_dy_2.shape[i]
      df_dy_dims_2[i] = df_dy_dims_1[i]
      out_dims += [df_dy_dims_1[i]]
    else:
      out_dims += [df_dy_dims_1[i], df_dy_dims_2[i]]

  return df_dy_dims_1, df_dy_dims_2, out_dims


def _vmap(f: Callable, in_axes, out_axes, squeeze_out: bool = True) -> Callable:
  """And expand-then-squeeze `vmap` for `f` expecting/returning batch dims."""
  in_axes_plus_1 = tree_map(lambda x: x if x in (None, -1) else x + 1, in_axes)

  @utils.wraps(f)
  def f_vmapped(*args):
    args = tree_map(_expand_dims, args, in_axes_plus_1,
                    is_leaf=lambda x: isinstance(x, np.ndarray))
    out = vmap(f, in_axes, out_axes)(*args)
    if squeeze_out:
      out_axes_plus_1 = tree_map(lambda x: x if x in (None, -1) else x + 1, out_axes)
      out = _squeeze(out, out_axes_plus_1)
    return out

  return f_vmapped


def _get_fx_axis_and_dtype(fx, fx_axis, params):
  if fx_axis is None:
    fx_axis = tree_map(
        lambda x: None, fx,
        is_leaf=lambda x: isinstance(x, (np.ndarray, utils.MaskedArray))
    )
  # Set the default type to be the least common type ancestor.
  dtypes, _ = tree_flatten(tree_map(np.dtype, params))
  if len(dtypes) == 0:
    dtype = None
  else:
    dtype = functools.reduce(np.promote_types, dtypes)
  return fx_axis, dtype


def _unravel_dfs(dfs, argnums, dyn_args, y):
  dfs = dfs[0] if isinstance(argnums, int) else dfs
  example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args

  dfs = tree_map(functools.partial(utils.unravel_array_into_pytree, y, 0), dfs)

  if tree_structure(dfs).num_leaves > 0:
    dfs = tree_transpose(tree_structure(tree_map(lambda x, y: [x] * len(y),
                                                 example_args,
                                                 dfs)),
                         tree_structure(y), dfs)

  if tree_structure(dfs).num_leaves == 0:
    dfs = tree_map(lambda x: dfs, y)
  return dfs


class _MODE(enum.Enum):
  """`F` - final output; `Y` - intermediary pre-activations; `W` - weights."""
  DF_DY = 'DF_DY'
  DY_DW = 'DY_DW'
  DF_DW = 'DF_DW'


def _get_df_dys_and_dy_dws(
    fun: Callable,
    fwd: Optional[bool],
    j_rules: bool,
    a_rules: bool,
    argnums: Union[int, Sequence[int]] = 0,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable:
  """Adapted from `jax.interpreters.ad`."""
  _check_callable(fun)

  def get_derivatives(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = jax.api_util.argnums_partial(f, argnums, args)
    tree_map(partial(
        jax._src.api._check_input_dtype_jacrev, holomorphic, allow_int),
        dyn_args)

    y_df_dy, pullback_df_dy = _vjp(
        f_partial,
        _MODE.DF_DY,
        fwd,
        j_rules,
        a_rules,
        *dyn_args)
    tree_map(partial(
        jax._src.api._check_output_dtype_jacrev, holomorphic),
        y_df_dy)

    y_dy_dw, pullback_dy_dw = _vjp(
        f_partial,
        _MODE.DY_DW,
        fwd,
        j_rules,
        a_rules,
        *dyn_args)
    tree_map(partial(
        jax._src.api._check_output_dtype_jacrev, holomorphic),
        y_dy_dw)

    dy_dws = pullback_dy_dw(y_dy_dw)
    dy_dws = dy_dws[0] if isinstance(argnums, int) else dy_dws

    df_dys = vmap(pullback_df_dy)(utils.std_basis(y_df_dy))
    df_dys = _unravel_dfs(df_dys, argnums, dyn_args, y_df_dy)
    return df_dys, dy_dws

  return get_derivatives


def _vjp(
    fun: lu.WrappedFun,
    mode: _MODE,
    fwd: Optional[bool],
    j_rules: bool,
    a_rules: bool,
    *primals,
    has_aux=False,
    reduce_axes=(),
):
  """Adapted from `jax.interpreters.ad`."""
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: jax._src.api._check_arg(arg)

  flat_fun, out_tree = jax._src.api.flatten_fun_nokwargs(fun, in_tree)
  outs = __vjp(
      flat_fun,
      mode=mode,
      primals=primals_flat,
      has_aux=has_aux,
      reduce_axes=reduce_axes,
      fwd=fwd,
      j_rules=j_rules,
      a_rules=a_rules,
  )

  if has_aux:
    out_primal, out_vjp, aux = outs
    out_tree, aux_tree = out_tree()
  else:
    out_primal, out_vjp = outs
    out_tree = out_tree()

  out_primal_py = tree_unflatten(out_tree, out_primal)
  ct_dtypes = [core.primal_dtype_to_tangent_dtype(jax._src.api._dtype(x))
               for x in out_primal]
  ct_shapes = [np.shape(x) for x in out_primal]

  vjp_py = Partial(partial(jax._src.api._vjp_pullback_wrapper,
                           ct_dtypes, ct_shapes,
                           (out_tree, in_tree)),
                   out_vjp)
  if has_aux:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)
  else:
    return out_primal_py, vjp_py


def __vjp(
    traceable,
    mode: _MODE,
    primals,
    has_aux,
    reduce_axes,
    fwd: Optional[bool],
    j_rules: bool,
    a_rules: bool
):
  """Adapted from `jax.interpreters.ad`."""
  with jax.disable_jit():
    outs = ad.linearize(traceable, *primals, has_aux=has_aux)

  if has_aux:
    out_primals, pvals, jaxpr, consts, aux = outs
  else:
    out_primals, pvals, jaxpr, consts = outs

  def unbound_vjp(pvals, jaxpr, consts, *cts):
    cts = tuple(map(ad.ignore_consts, cts, pvals))
    dummy_args = [UndefinedPrimal(v.aval) for v in jaxpr.invars]
    arg_cts = _backward_pass(jaxpr,
                             mode=mode,
                             reduce_axes=reduce_axes,
                             consts=consts,
                             primals_in=dummy_args,
                             cotangents_in=cts,
                             fwd=fwd,
                             j_rules=j_rules,
                             a_rules=a_rules)
    return map(ad.instantiate_zeros, arg_cts)

  vjp_ =  Partial(partial(unbound_vjp, pvals, jaxpr), consts)
  if has_aux:
    return out_primals, vjp_, aux
  else:
    return out_primals, vjp_


def _backward_pass(
    jaxpr: Jaxpr,
    mode: _MODE,
    reduce_axes,
    consts,
    primals_in,
    cotangents_in,
    fwd: Optional[bool],
    j_rules: bool,
    a_rules: bool
) -> List[List[Union[np.ndarray, Tuple[int, ...]]]]:
  """Adapted from `jax.interpreters.ad`."""
  if all(type(ct) is Zero for ct in cotangents_in):
    return map(lambda v: Zero(v.aval), jaxpr.invars)

  def read_cotangent(v):
    return ct_env.pop(v, Zero(v.aval))

  primal_env: Dict[Any, Any] = {}
  _write_primal(primal_env, core.unitvar, core.unit)
  map(partial(_write_primal, primal_env), jaxpr.constvars, consts)
  map(partial(_write_primal, primal_env), jaxpr.invars, primals_in)

  ct_env: Dict[Any, Any] = {}
  map(partial(_write_cotangent, 'outvars', ct_env, reduce_axes),
      jaxpr.outvars, cotangents_in)

  # List of `df_dy`s or `dy_dw`s for each variable in `jaxpr.invars`.
  outs = [[] for _ in jaxpr.invars]

  if mode in (_MODE.DY_DW, _MODE.DF_DW):
    invar_to_axes = rules.get_axes_cache(jaxpr, a_rules=a_rules)

  if mode == _MODE.DY_DW:
    vars_needing_cts_in = set()
  else:
    vars_needing_cts_in = _get_vars_needing_cts_in(jaxpr)

  for eqn in jaxpr.eqns[::-1]:
    # Do regular backprop.
    cts_in, invals = _backprop_step(
        eqn=eqn,
        primal_env=primal_env,
        ct_env=ct_env,
        read_cotangent=read_cotangent,
        reduce_axes=reduce_axes,
        do_write_cotangents=any(
            not isinstance(i, Literal) and i in vars_needing_cts_in
            for i in eqn.invars
        )
    )

    # Compute `df_dy`s or `dy_dw`s.
    for i_eqn, eq_invar in enumerate(eqn.invars):
      if eq_invar in jaxpr.invars:
        i_jaxpr = jaxpr.invars.index(eq_invar)
        inval = invals[i_eqn].aval

        if mode == _MODE.DF_DY:
          if eqn.primitive == lax.reshape_p:
            cts_in = cts_in.reshape(inval.shape)
          cts_in = cts_in.astype(inval.dtype)
          outs[i_jaxpr] += [cts_in]

        elif mode in (_MODE.DY_DW, _MODE.DF_DW):
          axes = rules.get_axes(eqn=eqn,
                                invals=[v.aval for v in eqn.invars],
                                idx=i_eqn,
                                a_rules=a_rules)
          axes &= invar_to_axes[eq_invar]

          if mode == _MODE.DY_DW:
            if eqn.primitive == lax.reshape_p:
              cts_in = ShapedArray(inval.shape, inval.dtype)
            elif hasattr(cts_in, 'aval'):
              cts_in = cts_in.aval

            trimmed_invals = _trim_invals(invals, axes)
            trimmed_cts_in = _trim_cotangents(cts_in, axes)
            eqn = _trim_eqn(eqn, i_eqn, trimmed_invals, trimmed_cts_in)

            def j_fn(invals):
              return _get_jacobian(eqn=eqn,
                                   cts_in=trimmed_cts_in,
                                   invals=invals,
                                   idx=i_eqn,
                                   reduce_axes=reduce_axes,
                                   fwd=fwd,
                                   j_rules=j_rules)

            for in_d, out_d in zip(axes.in_diagonal, axes.out_diagonal):
              in_axes = [
                  None
                  if isinstance(invals[ix], UndefinedPrimal)
                  else i
                  for ix, i in enumerate(in_d)]
              j_fn = _vmap(j_fn, in_axes=(in_axes,), out_axes=out_d)

            dy_dw = j_fn(trimmed_invals)
            outs[i_jaxpr] += [(dy_dw, axes)]

          elif mode == _MODE.DF_DW:
            trimmed_invals = _trim_invals(invals, axes)
            if eqn.primitive == lax.reshape_p:
              cts_in = cts_in.reshape(inval.shape)

            trimmed_cts_in = np.sum(cts_in, axes.out_broadcast, keepdims=True)
            eqn = _trim_eqn(eqn, i_eqn, trimmed_invals, trimmed_cts_in)

            def get_df_dw(cts_in, invals):
              return _eqn_vjp_fn(eqn, cts_in, reduce_axes, *invals)[i_eqn]

            for in_d, out_d in zip(axes.in_trace, axes.out_trace):
              in_axes = [None] * len(trimmed_invals)
              get_df_dw = _vmap(get_df_dw, (out_d, in_axes), in_d)

            for in_d, out_d in zip(axes.in_diagonal, axes.out_diagonal):
              in_axes = [
                  None
                  if isinstance(invals[ix], UndefinedPrimal)
                  else i
                  for ix, i in enumerate(in_d)]
              get_df_dw = _vmap(get_df_dw, (out_d, in_axes), in_d[i_eqn])

            df_dw = get_df_dw(trimmed_cts_in, trimmed_invals)
            outs[i_jaxpr] += [(df_dw, axes)]

          else:
            raise ValueError(mode)

        else:
          raise ValueError(mode)

  # If output contains input, this is not reflected in `jaxpr.eqns`.
  # Pass the `cotangents_in` as `df_dy`, and an identity matrix as `dy_dw`.
  for i_in, v_out in enumerate(jaxpr.outvars):
    for i_eqn, v in enumerate(jaxpr.invars):
      if v == v_out:
        if mode == _MODE.DF_DY:
          if v in ct_env:
            df_dy = cotangents_in[i_in]
          else:
            df_dy = v.aval

          outs[i_eqn] += [df_dy]
          break

        elif mode in (_MODE.DF_DW, _MODE.DY_DW):
          # Identity function
          axes = rules.get_id_axes(v.aval, a_rules)
          axes &= invar_to_axes[v]

          if mode == _MODE.DF_DW:
            if v in ct_env:
              df_dw = cotangents_in[i_in]
            else:
              df_dw = v.aval

            outs[i_eqn] += [(df_dw, axes)]
            break

          elif mode == _MODE.DY_DW:
            # Identity Jacobian
            trimmed_invals = _trim_invals([UndefinedPrimal(v.aval)], axes)
            trimmed_cts_in = _trim_cotangents(v.aval, axes)
            dy_dw = _get_jacobian(eqn=None,
                                  cts_in=trimmed_cts_in,
                                  invals=trimmed_invals,
                                  idx=0,
                                  j_rules=j_rules,
                                  fwd=fwd,
                                  reduce_axes=reduce_axes
                                  )
            outs[i_eqn] += [(dy_dw, axes)]

          else:
            raise ValueError(mode)

        else:
          raise ValueError(mode)

  return outs


def _get_vars_needing_cts_in(jaxpr: Jaxpr) -> Set[Var]:
  """Get a set of variables that need cotangents for structured derivatives."""
  need_cts: Set[Var] = set()

  def visit(vs: Set[Var]):
    if len(vs) == 0:
      return

    next_visit = set()

    for e in jaxpr.eqns:
      if any(v in e.invars for v in vs):
        for o in e.outvars:
          if o not in need_cts:
            need_cts.add(o)
            next_visit.add(o)

    visit(next_visit)

  visit(set(jaxpr.invars))

  # `invars` don't need cotangents in `STRUCTURED_DERIVATIVES` mode.
  assert all(i not in need_cts for i in jaxpr.invars)
  return need_cts


def _backprop_step(eqn,
                   primal_env,
                   ct_env,
                   read_cotangent,
                   reduce_axes,
                   do_write_cotangents: bool = True):
  """Adapted from `jax.interpreters.ad`."""
  invals = map(partial(_read_primal, primal_env), eqn.invars)
  cts_in = map(read_cotangent, eqn.outvars)
  if not eqn.primitive.multiple_results:
    cts_in = cts_in[0]
  else:
    raise NotImplementedError(len(eqn.outvars), eqn, cts_in)

  if do_write_cotangents:
    cts_out = _eqn_vjp_fn(eqn, cts_in, reduce_axes, *invals)
    cts_out = [Zero(v.aval) for v in eqn.invars] if cts_out is Zero else cts_out
    map(partial(_write_cotangent, eqn.primitive, ct_env, reduce_axes),
        eqn.invars, cts_out)
  return cts_in, invals


def _trim_cotangents(cts_in: ShapedArray, axes: rules.Axes) -> ShapedArray:
  cts_in = _trim_axis(cts_in,
                      axes.out_trace + axes.out_broadcast + axes.out_diagonal)
  return cts_in


def _trim_invals(invals: List[np.ndarray],
                 axes: rules.Axes,
                 shift: int = 0) -> List:
  trimmed_invals = list(invals)

  for i in axes.in_trace_idxs:
    trimmed_invals[i] = _trim_axis(trimmed_invals[i], axes.in_trace, shift)

  for ax in axes.in_broadcast:
    for ix, i in enumerate(ax):
      if i is not None:
        trimmed_invals[ix] = _trim_axis(trimmed_invals[ix], i, shift)

  for i in range(len(trimmed_invals)):
    for in_d in sorted([axis[i] for axis in axes.in_diagonal
                        if axis[i] is not None],
                       reverse=True):
      if isinstance(trimmed_invals[i], UndefinedPrimal):
        trimmed_invals[i] = _trim_axis(trimmed_invals[i], in_d, shift)

  return trimmed_invals


def _trim_eqn(eqn: JaxprEqn,
              idx: int,
              trimmed_invals: List[np.ndarray],
              trimmed_cts_in: ShapedArray,
              shift: int = 0) -> JaxprEqn:
  out = JaxprEqn(
      invars=eqn.invars,
      outvars=eqn.outvars,
      primitive=eqn.primitive,
      source_info=eqn.source_info,
      params=dict(eqn.params)
  )

  if out.primitive == lax.broadcast_in_dim_p:
    # `broadcast_in_dim` is the only primitive JVP where we need to change
    # equation parameters in response to tweaking the inputs/cotangents
    # shapes.
    out.params['shape'] = trimmed_cts_in.shape[shift:]

  elif out.primitive == lax.reshape_p:
    # Hack for more efficient `reshape` axes rules.
    out.params['new_sizes'] = trimmed_invals[idx].aval.shape

  elif out.primitive == lax.conv_general_dilated_p:
    if idx == 0:
      out.params['lhs_shape'] = trimmed_invals[0].aval.shape
      out.params['rhs_shape'] = trimmed_invals[1].shape[max(shift - 1, 0):]
    elif idx == 1:
      out.params['lhs_shape'] = trimmed_invals[0].shape[max(shift - 1, 0):]
      out.params['rhs_shape'] = trimmed_invals[1].aval.shape
    else:
      raise ValueError(out, idx)

  return out


def _trim_axis(
    x: Union[UndefinedPrimal, ShapedArray, np.ndarray],
    axis: Union[int, Tuple[int, ...]],
    shift: int = 0
) -> Union[UndefinedPrimal, ShapedArray]:
  """Trim `axis` of `x` to be of length `1`. `x` is only used for shape."""
  if isinstance(axis, int):
    axis = (axis,)

  if isinstance(x, UndefinedPrimal):
    return UndefinedPrimal(_trim_axis(x.aval, axis))

  if isinstance(x, (ShapedArray, np.ndarray)):
    return ShapedArray([1 if i - shift in axis else x.shape[i - shift]
                        for i in range(shift, x.ndim)], dtype=x.dtype)

  raise TypeError(type(x), x)


def _eqn_jvp_fn(eqn: Optional[JaxprEqn],
                idx: int,
                tangents: np.ndarray,
                *invals):
  if eqn is None:
    # Identity function
    return tangents

  new_tangents = []
  new_invals = []

  for i_dx, i in enumerate(invals):
    if i_dx == idx:
      inval = np.zeros(i.aval.shape, i.aval.dtype)
      tangent = tangents
    else:
      inval = i
      aval = i.aval if hasattr(i, 'aval') else ShapedArray(i.shape, i.dtype)
      tangent = Zero(aval)
      if isinstance(inval, (UndefinedPrimal, ShapedArray)):
        inval = np.zeros(aval.shape, aval.dtype)

    new_invals.append(inval)
    new_tangents.append(tangent)

  jvp_fn = ad.primitive_jvps[eqn.primitive]
  return jvp_fn(new_invals, new_tangents, **eqn.params)[1]


def _eqn_vjp_fn(eqn: Optional[JaxprEqn],
                cts_in: np.ndarray,
                reduce_axes,
                *invals):
  """Adapted from `jax.interpreters.ad`."""
  if eqn is None:
    # Identity function
    return (cts_in,)

  with ad.source_info_util.user_context(eqn.source_info):
    if eqn.primitive.call_primitive or eqn.primitive.map_primitive:
      cts_in_avals = [v.aval for v in eqn.outvars]
      call_jaxpr, params = core.extract_call_jaxpr(eqn.primitive, eqn.params)
      cts_out = ad.get_primitive_transpose(eqn.primitive)(
          params, call_jaxpr, invals, cts_in, cts_in_avals, reduce_axes)
    elif eqn.primitive in ad.reducing_transposes:
      cts_out = ad.reducing_transposes[eqn.primitive](
          reduce_axes, cts_in, *invals, **eqn.params)
    else:
      cts_out = ad.get_primitive_transpose(eqn.primitive)(cts_in, *invals,
                                                          **eqn.params)
  return cts_out


def _get_jacobian(
    eqn: Optional[JaxprEqn],
    cts_in: ShapedArray,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    idx: int,
    j_rules: bool,
    fwd: Optional[bool],
    reduce_axes
):
  if eqn is None:
    primitive = None
  else:
    primitive = eqn.primitive

  inval_shape = invals[idx].aval.shape
  cts_in_shape = cts_in.shape

  if primitive == xla.xla_call_p:
    raise NotImplementedError(eqn)

  if primitive not in rules.JACOBIAN_RULES:
    warnings.warn(f'No Jacobian rule found for {primitive}.')

  if primitive in rules.JACOBIAN_RULES and j_rules:
    # Custom Jacobian rule.
    dy_dw = rules.JACOBIAN_RULES[primitive](eqn, idx, invals, cts_in)

  else:
    # Vanilla Jacobian evaluation.
    if _get_fwd(fwd, cts_in_shape, inval_shape):
      # Forward mode.
      out_axes = -1
      inputs = invals[idx].aval
      def jac_fn(tangents):
        return _eqn_jvp_fn(eqn, idx, tangents, *invals)

    else:
      # Reverse mode.
      out_axes = 0
      inputs = cts_in
      def jac_fn(cotangents):
        return _eqn_vjp_fn(eqn, cotangents, reduce_axes, *invals)[idx]

    eye = utils.std_basis(inputs)
    dy_dw = vmap(jac_fn, out_axes=out_axes)(eye)
    dy_dw = dy_dw.reshape(cts_in_shape + inval_shape)

  assert dy_dw.shape == cts_in_shape + inval_shape, (dy_dw.shape, cts_in_shape, inval_shape)
  return dy_dw


def _write_cotangent(prim, ct_env, reduce_axes, v, ct):
  """Adapted from `jax.interpreters.ad`."""
  assert ct is not Zero, (prim, v.aval)
  if ct is None or type(v) is Literal:
    return
  if type(ct) is Zero:
    return
  axes_to_reduce = tuple(axis_name for axis_name in reduce_axes
                         if axis_name in core.get_aval(ct).named_shape
                         and axis_name not in v.aval.named_shape)
  if axes_to_reduce:
    ct = lax.psum(ct, axis_name=axes_to_reduce)
  ct_env[v] = ad.add_tangents(ct_env[v], ct) if v in ct_env else ct
  if ad.config.jax_enable_checks:
    ct_aval = core.get_aval(ct_env[v])
    joined_aval = core.lattice_join(
        v.aval, ct_aval).strip_weak_type().strip_named_shape()
    assert v.aval.strip_weak_type().strip_named_shape() == joined_aval, (prim, v.aval, ct_aval)


def _read_primal(env, v, str_match: bool = False):
  if type(v) is Literal:
    return v.val

  if v is core.unitvar:
    return core.unit

  if v in env:
    return env[v]

  if str_match:
    for v_ in env:
      if str(v) == str(v_):
        return env[v_]

  return UndefinedPrimal(v.aval)


def _write_primal(env, v, val):
  if not ad.is_undefined_primal(val):
    env[v] = val


def _get_fwd(fwd: Optional[bool], cts_in_shape, inval_shape) -> bool:
  if fwd is None:
    out_size = onp.prod(cts_in_shape)
    in_size = onp.prod(inval_shape)
    fwd = out_size > in_size
  return fwd


def jacobian_calculator(f: ApplyFn, vmap_axes: VMapAxes = None) \
  -> Callable[[NTTree[np.ndarray], Optional[NTTree[np.ndarray]], PyTree], NTTree[np.ndarray]]:

  def _jacobian_calc(x: NTTree[np.ndarray], params: PyTree = None) -> np.ndarray:

    fx = eval_shape(f, params, x)
    x_axis, fx_axis, kw_axes = _canonicalize_axes(vmap_axes, x, fx)

    def j_fn(x):
      fx = _get_f_params(f, x, x_axis, fx_axis, kw_axes)
      jx = jacobian(fx)(params)
      return jx

    if x_axis is not None or kw_axes:
      in_axes = [x_axis]
      j_fn = vmap(j_fn, in_axes=in_axes, out_axes=fx_axis)

    j = j_fn(x)
    j = np.concatenate(tree_flatten(tree_map(lambda x: x.reshape(*fx.shape, -1), j))[0], axis=2)
    return j

  return _jacobian_calc