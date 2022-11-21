"""General-purpose internal utilities."""

import functools
import inspect
import operator
from typing import Any, Callable, Iterable, List, Optional, Sequence, Sized, Tuple, Union
from neural_tangents.utils.typing import Axes, PyTree
import warnings

from neural_tangents.utils import dataclasses
import jax
from jax import lax, dtypes, tree_flatten, tree_unflatten
from jax.core import ShapedArray
from jax import random
from jax.lib import xla_bridge
import jax.numpy as np
from jax.tree_util import tree_all, tree_map
import numpy as onp


def is_list_or_tuple(x):
  # We do not want to return True if x is a subclass of list or tuple since
  # otherwise this will return true for namedtuples.
  return type(x) == list or type(x) == tuple


def is_nt_tree_of(x, dtype):
  if isinstance(x, dtype):
    return True
  if not is_list_or_tuple(x):
    return False
  return all(is_nt_tree_of(_x, dtype) for _x in x)


def nt_tree_fn(nargs: Optional[int] = None,
               tree_structure_argnum: Optional[int] = None,
               reduce: Callable = lambda x: x):
  """Convert a function that acts on single inputs to one that acts on trees.

  `nt_tree_fn` treats the first `nargs` arguments as NTTrees and the remaining
  arguments as broadcasted over the tree structure. `nt_tree_fn` then calls the
  function on each leaf of the tree. Each node of the tree optionally calls a
  reduce function over the values of its children.

  If `tree_structure_argnum` is None then each of the NTTrees must have the same
  structure. If `tree_structure_argnum` is an integer then then a specific tree
  is used to infer the structure.

  Args:
    nargs: The number of arguments to be treated as NTTrees. If `nargs` is None
      then all of the arguments are used. `nargs` can also be negative which
      follows numpy's semantics for array indexing.
    tree_structure_argnum: The argument used to infer the tree structure to be
      traversed. If `tree_structure_argnum` is None then a check is performed to
      ensure that all trees have the same structure.
    reduce: A callable that is applied recursively by each internal tree node
      to its children.

  Returns:
    A decorator `tree_fn` that transforms a function, `fn`, from acting on
    leaves to acting on NTTrees.
  """

  def check_tree_structure(args):
    """Ensure the structure of the trees in each of the `nargs` is the same."""
    if any(is_list_or_tuple(x) for x in args):
      if not all(type(x) == type(args[0]) for x in args[1:]):
        raise TypeError(f'Inconsistent NTTree structure found. '
                        f'Node Types: {[type(x) for x in args]}.')

      """
        Regarding the use of zip, consider an example `x1 = x2 = (1, (1, 1))`.
        We would like to determine whether these two trees have the same
        structure.

        On the first recurrence `x1` and `x2` are both tuples so the check
        passes and `zip(*args) = [(1, 1), ((1, 1), (1, 1))]` so that
        `(check_tree_structure(x) for x in zip(x1, x2))` will first check that
        the first element of `x1` has the same tree structure as the first
        element of `x2` and then the second element and so on.
      """
      for x in zip(*args):
        check_tree_structure(x)

  def tree_fn(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
      _nargs = len(args) if nargs is None else nargs
      recurse, norecurse = args[:_nargs], args[_nargs:]

      structure_argnum = tree_structure_argnum
      if structure_argnum is None:
        check_tree_structure(recurse)
        structure_argnum = 0

      if is_list_or_tuple(args[structure_argnum]):
        list_or_tuple = type(args[structure_argnum])
        return reduce(list_or_tuple(
            wrapped_fn(*(xs + norecurse), **kwargs) for xs in zip(*recurse)))
      return fn(*args, **kwargs)
    return wrapped_fn
  return tree_fn


def all_none(x: Any, attr: Optional[str] = None) -> bool:
  get_fn = (lambda x: x) if attr is None else lambda x: getattr(x, attr)
  return tree_all(tree_map(lambda x: get_fn(x) is None, x))


def wraps(f):
  def wrapper(g):
    @functools.wraps(f)
    def h(*args, **kwargs):
      return g(*args, **kwargs)

    h.__signature__ = inspect.signature(f)
    return h
  return wrapper


@nt_tree_fn(nargs=2, reduce=lambda x: np.all(np.array(x)))
def x1_is_x2(x1: np.ndarray,
             x2: Optional[np.ndarray] = None,
             eps: float = 1e-12) -> Union[bool, np.ndarray]:
  if not isinstance(x1, (onp.ndarray, np.ndarray)):
    raise TypeError('`x1` must be an ndarray. A {} is found.'.format(type(x1)))

  if x2 is None:
    return True

  if x1 is x2:
    return True

  if x1.shape != x2.shape:
    return False

  if xla_bridge.get_backend().platform == 'tpu':
    eps = 1e-4

  return np.all(np.abs(x1 - x2) < eps)


def _get_ndim(x: Union[int, Sized, np.ndarray]) -> int:
  """Get number of dimensions given number of dimensions / shape / array."""
  if hasattr(x, 'ndim'):
    n = x.ndim
  elif hasattr(x, '__len__'):
    n = len(x)
  elif isinstance(x, int):
    n = x
  else:
    raise TypeError(x, type(x))
  return n


def canonicalize_axis(axis: Axes,
                      x: Union[int, Sized, np.ndarray]) -> List[int]:
  """Converts axis into a sorted non-negative list.

  Args:
    axis: input axis.
    x: array / shape / number of dimensions.

  Returns:
    A sorted list of integer axes.
  """
  axis = [axis] if isinstance(axis, int) else list(axis)
  n = _get_ndim(x)
  return list(set(onp.arange(n)[axis]))


def zip_axes(x: np.ndarray,
             start_axis: int = 0,
             end_axis: Optional[int] = None) -> np.ndarray:
  """Zip (interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip (interleave).
    end_axis: `int`, number of axis until which to zip (interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=False)


def unzip_axes(x: np.ndarray,
               start_axis: int = 0,
               end_axis: Optional[int] = None) -> np.ndarray:
  """Unzip (de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
  `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ...]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to unzip (de-interleave).
    end_axis: `int`, number of axis until which to unzip (de-interleave).

  Returns:
    A `np.ndarray` with a new shape.
  """
  return _zip_axes(x, start_axis, end_axis, unzip=True)


def _zip_axes(x: np.ndarray,
              start_axis: int = 0,
              end_axis: Optional[int] = None,
              unzip: bool = False) -> np.ndarray:
  """Zip/unzip (interleave/de-interleave) axes starting from `start_axis`.

  Changes the shape as follows:
    If `unzip == True`:
    `[..., X, X, ..., Y, Y, ..., Z, Z, ...] -> [..., X, Y, Z, ..., X, Y, Z, ..]`
    If `unzip == False`:
    `[..., X, Y, Z, ..., X, Y, Z, ...] -> [..., X, X, ..., Y, Y, ..., Z, Z, ..]`

  Args:
    x: `np.ndarray` with an even number of dimensions following `start_axis`.
    start_axis: `int`, number of axis from which to zip/unzip.
    end_axis: `int`, number of axis until which to zip/unzip.
    unzip: `bool`, set to `True` to unzip instead of zip.

  Returns:
    A `np.ndarray` with a new shape.
  """
  if end_axis is None:
    end_axis = x.ndim

  half_ndim, ragged = divmod(end_axis - start_axis, 2)
  if ragged:
    raise ValueError(
        f'Need even number of axes to zip, got {end_axis - start_axis}.')

  odd_axes = range(start_axis + 1, end_axis, 2)
  last_axes = range(end_axis - half_ndim, end_axis)

  if unzip:
    x = np.moveaxis(x, odd_axes, last_axes)
  else:
    x = np.moveaxis(x, last_axes, odd_axes)
  return x


@dataclasses.dataclass
class MaskedArray:
  masked_value: Union[np.ndarray, float]
  mask: np.ndarray
  shape: Tuple[int, ...] = dataclasses.field(init=False, pytree_node=False)
  ndim: int = dataclasses.field(init=False, pytree_node=False)

  def __post_init__(self):
    if isinstance(self.masked_value, (float, int)):
      shape = ()
      ndim = 0
      dtype = dtypes.canonicalize_dtype(type(self.masked_value))

    else:
      # pytype:disable=attribute-error
      shape = self.masked_value.shape
      ndim = self.masked_value.ndim
      dtype = self.masked_value.dtype
      # pytype:enable=attribute-error

    super().__setattr__('shape', shape)
    super().__setattr__('ndim', ndim)
    super().__setattr__('dtype', dtype)

  astuple = ...  # type: Callable[[], Tuple[np.ndarray, np.ndarray, Tuple[int, ...], int]]


@nt_tree_fn(nargs=1)
def get_masked_array(x: Union[ShapedArray, np.ndarray],
                     mask_constant: Optional[float] = None) -> MaskedArray:
  """Return `x` with entries equal to `mask_constant` zeroed-out, and the mask.

  The mask returned is a boolean `np.ndarray` with masked indices having `True`.

  Args:
    x: `np.ndarray` to mask. If `x` is a `MaskedInput`, treat it as
      `(masked_x, mask)` and pass it through.
    mask_constant: an optional `float`, the value in inputs to be considered as
      masked (e.g. padding in a batch of sentences). `None` means no masking.
      Can also be `np.nan`, `np.inf` etc.

  Returns:
    A `MaskedArray` of `(masked_x, boolean_mask)`.
  """

  if x is None:
    mask_mat = None

  elif isinstance(x, MaskedArray):
    x, mask_mat, _, _ = x.astuple()

  elif isinstance(x, (onp.ndarray, np.ndarray, float, int)):
    if mask_constant is None:
      mask_mat = None
    else:
      mask_mat = lax.cond(np.isnan(mask_constant),
                          np.isnan,
                          lambda x: x == mask_constant,
                          x)
  else:
    raise TypeError(x, type(x))

  x = mask(x, mask_mat)
  return MaskedArray(x, mask_mat)  # pytype: disable=wrong-arg-count


def mask(x: Optional[np.ndarray], mask_mat: Optional[np.ndarray]):
  if x is None or mask_mat is None:
    return x
  return np.where(mask_mat, np.zeros((), x.dtype), x)


def size_at(x: Union[np.ndarray, Sequence[int], ShapedArray],
            axes: Optional[Iterable[int]] = None) -> int:
  if hasattr(x, 'aval'):
    x = x.aval

  if hasattr(x, 'shape'):
    x = x.shape

  if axes is None:
    axes = range(len(x))

  return functools.reduce(operator.mul, [x[a] for a in axes], 1)


def get_res_batch_dims(contracting_dims: Iterable[int],
                       batch_dims: Iterable[int]) -> List[int]:
  res_batch_dims = [2 * b - i for i, b in enumerate(batch_dims)]
  for i, b in enumerate(batch_dims):
    for c in contracting_dims:
      if b > c:
        res_batch_dims[i] -= 2
  return res_batch_dims


def dot_general(lhs: np.ndarray,
                rhs: np.ndarray,
                contracting_dims: Axes,
                batch_dims: Axes,
                precision=None) -> np.ndarray:
  """`jax.lax.dot_general` with preserved dims order and shared lhs / rhs dims.

  Precisely, returns `jax.lax.dot_general(lhs, rhs, dimension_numbers)` where
  `dimension_numbers == ((contracting_dims, contracting_dims),
                         (batch_dims, batch_dims))`,
  but preserves the dimension order in the output. See XLA's
   `DotGeneral<https://www.tensorflow.org/xla/operation_semantics#dotgeneral>`.

  Args:
    lhs: array.
    rhs: array, must have the same dimensionality as `lhs`.
    contracting_dims: contracting dimensions.
    batch_dims: batch dimensions.
    precision: Optional. Either `None`, which means the default precision for
      the backend, or a `Precision` enum value.

  Returns:
    Dot product result with preserved dimension order.
  """
  if lhs.ndim != rhs.ndim:
    raise ValueError(f'`lhs` and `rhs` must have the same dimensionality, got'
                     f'`lhs.ndim == {lhs.ndim}` and `rhs.ndim == {rhs.ndim}`.')

  contracting_dims = canonicalize_axis(contracting_dims, lhs)
  batch_dims = canonicalize_axis(batch_dims, lhs)

  n_batch_dims = len(batch_dims)
  leading_batch_dims = range(n_batch_dims)

  dimension_numbers = ((contracting_dims, contracting_dims),
                       (batch_dims, batch_dims))

  prod = lax.dot_general(lhs, rhs, dimension_numbers, precision)
  prod = zip_axes(prod, n_batch_dims)

  res_batch_dims = get_res_batch_dims(contracting_dims, batch_dims)
  prod = np.moveaxis(prod, leading_batch_dims, res_batch_dims)
  return prod


def axis_after_dot(axis: int,
                   contracting_dims: Sequence[int],
                   batch_dims: Sequence[int],
                   lhs_ndim: Optional[int] = None) -> int:
  if axis in batch_dims:
    return batch_dims.index(axis)

  return (
      axis -
      sum(1 for i in contracting_dims if i < axis) +
      sum(1 for i in batch_dims if i > axis) +
      (0 if lhs_ndim is None
       else lhs_ndim - len(batch_dims) - len(contracting_dims))
  )


def _read_keys(key, x1, x2):
  """Read dropout key.

  `key` might be a tuple of two rng keys or a single rng key or None. In
  either case, `key` will be mapped into two rng keys `key1` and `key2` to
  make sure `(x1==x2) == (key1==key2)`.
  """

  if key is None or all_none(x2):
    key1 = key2 = key
  elif isinstance(key, tuple) and len(key) == 2:
    key1, key2 = key
    new_key = np.where(x1_is_x2(key1, key2),
                       random.fold_in(key2, 1), key2)
    key2 = np.where(x1_is_x2(x1, x2), key1, new_key)
    warnings.warn('The value of `key[1]` might be replaced by a new value if '
                  'key[0] == key[1] and x1 != x2 or key[0] != key[1] and '
                  'x1 == x2.')
  elif isinstance(key, (onp.ndarray, np.ndarray)):
    key1 = key
    key2 = np.where(x1_is_x2(x1, x2), key1, random.fold_in(key, 1))
  else:
    raise TypeError(type(key))
  return key1, key2


def split_kwargs(kwargs, x1=None, x2=None):
  """Splitting `kwargs`.

     Specifically,
       1. if kwarg is an rng key, it will be split into two keys.
       2. else if it is a tuple of length two, the tuple will be split into two
          parts, one for kwargs1 and the other for kwargs2.
       3. else it is copied to kwargs1 and kwargs2.

  """
  kwargs1 = {}
  kwargs2 = {}
  for k, v in kwargs.items():
    if x2 is not None and k == 'rng':
      key1, key2 = _read_keys(v, x1, x2)
      kwargs1[k] = key1
      kwargs2[k] = key2
    elif isinstance(v, tuple) and len(v) == 2:
      kwargs1[k] = v[0]
      kwargs2[k] = v[1]
    else:
      kwargs1[k] = kwargs2[k] = v

  return kwargs1, kwargs2


def get_flops(f: Callable, optimize: bool, *a, **kw) -> float:
  m = jax.xla_computation(f)(*a, **kw)
  client = jax.lib.xla_bridge.get_backend()
  if optimize:
    m = client.compile(m).hlo_modules()[0]
  else:
    m = m.as_hlo_module()
  analysis = jax.lib.xla_client._xla.hlo_module_cost_analysis(client, m)

  if 'flops' not in analysis:
    warnings.warn('No `"flops"` returned by HLO cost analysis.')
    return onp.inf

  return analysis['flops']


def std_basis(pytree: PyTree) -> PyTree:
  """Same as `jax.api._std_basis` but without host-side ops."""
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(np.size, leaves))
  dtype = dtypes.result_type(*leaves)
  flat_basis = np.eye(ndim, dtype=dtype)
  return unravel_array_into_pytree(pytree, 1, flat_basis)


def unravel_array_into_pytree(pytree: PyTree,
                              axis: int,
                              arr: np.ndarray) -> PyTree:
  """Same as `jax.api._unravel_array_into_pytree but without host-side ops."""
  leaves, treedef = tree_flatten(pytree)
  if arr.ndim > 0:
    axis %= arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = np.split(arr, onp.cumsum([np.size(l) for l in leaves[:-1]]), axis)
  reshaped_parts = [np.reshape(x, shape) for x, shape in zip(parts, shapes)]
  return tree_unflatten(treedef, reshaped_parts)


def squeeze(x: np.ndarray,
            axis: Union[None, int, Tuple[int, ...]]) -> np.ndarray:
  """`np.squeeze` analog working with 0-sized axes."""
  if isinstance(axis, int):
    axis = (axis,)

  non_zero_axes = tuple()
  shift = 0

  for a in sorted(axis):
    if x.shape[a - shift] == 0:
      new_shape = x.shape[:a] + x.shape[a + 1:]
      if size_at(new_shape) == 0:
        x = x.reshape(new_shape)
      else:
        x = np.zeros(new_shape, x.dtype)

      shift += 1
    else:
      non_zero_axes += (a - shift,)

  return np.squeeze(x, non_zero_axes)