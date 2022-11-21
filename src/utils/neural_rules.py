"""Structured derivatives rules."""

import functools
from typing import Callable, Optional, Tuple, Dict, List, Union
import jax.numpy as np
import jax
from jax import lax
from jax.interpreters import ad, xla
import numpy as onp
from jax.interpreters.ad import UndefinedPrimal
from jax.core import JaxprEqn, ShapedArray, Primitive, Jaxpr, Var, AbstractValue, Literal
from src.utils import neural_utils as utils
from neural_tangents.utils.dataclasses import dataclass, field


# pytype: disable=wrong-keyword-args
# pytype: disable=attribute-error


@dataclass
class Axes:
  """Describes structure present in a primitive derivative dy/dtheta."""
  out_trace: Tuple[int, ...] = field(False, default_factory=tuple)
  in_trace: Tuple[int, ...] = field(False, default_factory=tuple)
  in_trace_idxs: Tuple[int, ...] = field(False, default_factory=tuple)

  out_diagonal: Tuple[int, ...] = field(False, default_factory=tuple)
  in_diagonal: Tuple[Tuple[Optional[int], ...], ...] = field(
      False, default_factory=tuple)

  out_broadcast: Tuple[int, ...] = field(False, default_factory=tuple)
  in_broadcast: Tuple[Tuple[Optional[int], ...], ...] = field(
      False, default_factory=tuple)

  def __and__(self, other):
    """Defines interaction with structure of the other primitive dy2/dtheta."""
    assert len(self.in_trace) == len(self.out_trace)
    assert len(other.in_trace) == len(other.out_trace)

    in_trace_idxs = self.in_trace_idxs
    in_trace = tuple(i for i in self.in_trace if i in other.in_trace)

    out_trace = tuple(self.out_trace[i] for i in range(len(self.out_trace))
                      if self.in_trace[i] in other.in_trace
                      )

    assert len(in_trace) == len(out_trace)

    out_diagonal = tuple(i for i in self.out_diagonal
                         if i in other.out_diagonal)
    in_diagonal = tuple(i for ix, i in enumerate(self.in_diagonal)
                        if self.out_diagonal[ix] in other.out_diagonal)

    out_broadcast = tuple(i for i in self.out_broadcast
                          if i in other.out_broadcast)
    in_broadcast = tuple(i for ix, i in enumerate(self.in_broadcast)
                         if self.out_broadcast[ix] in other.out_broadcast)

    return Axes(
        out_trace=out_trace,
        in_trace=in_trace,
        in_trace_idxs=in_trace_idxs,
        out_diagonal=out_diagonal,
        in_diagonal=in_diagonal,
        out_broadcast=out_broadcast,
        in_broadcast=in_broadcast,
    )


AXES_RULES : Dict[Optional[Primitive], Callable[..., Axes]] = {}
JACOBIAN_RULES: Dict[Optional[Primitive], Callable[..., np.ndarray]] = {}


def get_axes(
    eqn: Optional[JaxprEqn],
    invals: List[ShapedArray],
    idx: int,
    a_rules: bool
) -> Axes:
  if eqn is None:
    # Identity function
    primitive = None
    cts_in = invals[0]
    assert idx == 0

  else:
    if len(eqn.outvars) != 1:
      raise NotImplementedError(eqn)
    cts_in = eqn.outvars[0].aval

    primitive = eqn.primitive
    assert len(invals) == len(eqn.invars)
    assert 0 <= idx < len(eqn.invars)

  if not isinstance(cts_in, ShapedArray):
    raise TypeError(cts_in)

  if primitive in AXES_RULES and a_rules:
    axes = AXES_RULES[primitive](eqn, idx, invals, cts_in)

  else:
    # No simplification rule found.
    axes = Axes()

  if primitive == lax.reshape_p:
    cts_in = ShapedArray(invals[idx].shape, invals[idx].dtype)

  # Check that number of trace output and input axes match.
  assert len(axes.in_trace) == len(axes.out_trace)

  # Check that input and output traced sizes are the same.
  out_trace_size = utils.size_at(cts_in, axes.out_trace)
  in_trace_size = utils.size_at(invals[idx], axes.in_trace)
  assert in_trace_size == out_trace_size

  # Check that number of input/output diagonal/broadcast axes match.
  assert len(axes.out_diagonal) == len(axes.in_diagonal)
  assert len(axes.out_broadcast) == len(axes.in_broadcast)

  # Check for each output diagonal axis there's only input axes of correct
  # size or `None`. Inval axis should be not `None`.
  for out_d, in_d in zip(axes.out_diagonal, axes.in_diagonal):
    assert len(in_d) == len(invals)
    assert in_d[idx] is not None
    for ix, i in enumerate(in_d):
      if i is not None:
        assert invals[ix].shape[i] == cts_in.shape[out_d]

  # Check for each output broadcast axis there's only input axes of correct
  # size or `None`. Inval axis should be `None`.
  for out_d, in_d in zip(axes.out_broadcast, axes.in_broadcast):
    assert len(in_d) == len(invals)
    assert in_d[idx] is None
    for ix, i in enumerate(in_d):
      if i is not None:
        assert invals[ix].shape[i] == cts_in.shape[out_d]

  return axes


def get_axes_cache(
    jaxpr: Jaxpr,
    a_rules: bool
) -> Dict[Var, Axes]:
  invar_to_axes: Dict[Var, Axes] = {}

  for var in jaxpr.invars:
    if var in jaxpr.outvars:
      if isinstance(var, Literal):
        raise TypeError(var)

      # Identity function
      axes = get_id_axes(var.aval, a_rules)

      if var in invar_to_axes:
        invar_to_axes[var] &= axes
      else:
        invar_to_axes[var] = axes

  for eqn in jaxpr.eqns:
    for i_eqn, var in enumerate(eqn.invars):
      if var in jaxpr.invars:
        if isinstance(var, Literal):
          raise TypeError(var)

        axes = get_axes(eqn=eqn,
                        invals=[v.aval for v in eqn.invars],
                        idx=i_eqn,
                        a_rules=a_rules)

        if var in invar_to_axes:
          invar_to_axes[var] &= axes
        else:
          invar_to_axes[var] = axes

  return invar_to_axes


def get_id_axes(
    inval: AbstractValue,
    a_rules: bool
) -> Axes:
  if not isinstance(inval, ShapedArray):
    raise TypeError(inval)

  eqn = None
  idx = 0
  invals = [inval]
  return get_axes(eqn, invals, idx, a_rules)


# UTILS


def _eye_like(out_shaped: ShapedArray, in_shaped: ShapedArray) -> np.ndarray:
  assert out_shaped.size == in_shaped.size, (out_shaped, in_shaped)
  eye = np.eye(out_shaped.size, dtype=in_shaped.dtype)
  eye = eye.reshape(out_shaped.shape + in_shaped.shape)
  return eye


# BINARY PRIMITIVES


def _dot_general_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  contracting_dims, batch_dims = eqn.params['dimension_numbers']
  self, other = invals[idx], invals[1 if idx == 0 else 0]

  self_c_dims = contracting_dims[idx]

  self_b_dims = batch_dims[idx]

  in_trace = tuple(i for i in range(self.ndim) if
                   (i not in self_c_dims) and (i not in self_b_dims))
  out_trace = tuple(
    utils.axis_after_dot(i, self_c_dims, self_b_dims,
                         lhs_ndim=None if idx == 0 else other.ndim)
    for i in in_trace
  )

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              in_diagonal=tuple(zip(*batch_dims)),
              out_diagonal=tuple(range(len(self_b_dims)))
              )

def _dot_general_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  contracting_dims, batch_dims = eqn.params['dimension_numbers']

  lhs_c_dims, rhs_c_dims = contracting_dims
  lhs_b_dims, rhs_b_dims = batch_dims

  lhs, rhs = invals

  if idx == 0:
    self = lhs.aval
    self_c_dims, self_b_dims = lhs_c_dims, lhs_b_dims

    other = rhs
    other_c_dims, other_b_dims = rhs_c_dims, rhs_b_dims

  else:
    self = rhs.aval
    self_c_dims, self_b_dims = rhs_c_dims, rhs_b_dims

    other = lhs
    other_c_dims, other_b_dims = lhs_c_dims, lhs_b_dims

  self_ncb_dims = tuple(i for i in range(self.ndim)
                        if i not in self_c_dims + self_b_dims)
  self_nc_dims = tuple(i for i in range(self.ndim)
                       if i not in self_c_dims)

  j = np.moveaxis(
      other,
      other_b_dims + other_c_dims,
      tuple(range(len(other_b_dims))) + tuple(range(-len(other_c_dims), 0))
  )

  self_ncb_out = tuple(utils.axis_after_dot(
      i,
      self_c_dims,
      self_b_dims,
      other.ndim if idx == 1 else None
  ) for i in self_ncb_dims)

  self_nc_in = tuple(cts_in.ndim + i for i in self_nc_dims)
  j = np.expand_dims(j, self_ncb_out + self_nc_in)

  self_ncb_size = utils.size_at(self, self_ncb_dims)
  self_ncb_in = tuple(i + cts_in.ndim for i in self_ncb_dims)
  shape = [1 for _ in range(j.ndim)]
  for i_out, i_in in zip(self_ncb_out, self_ncb_in):
    shape[i_out] = shape[i_in] = self.shape[i_in - cts_in.ndim]

  eye = np.eye(self_ncb_size, dtype=np.bool_)
  eye = eye.reshape(shape)
  j = np.where(eye, j, np.zeros((), j.dtype))

  for out_b, (self_b, other_b) in enumerate(zip(self_b_dims, other_b_dims)):
    b_size = other.shape[other_b]
    eye = np.eye(b_size, dtype=np.bool_)
    shape = [1 for _ in range(j.ndim)]
    shape[out_b] = shape[cts_in.ndim + self_b] = b_size
    eye = eye.reshape(shape)
    j = np.where(eye, j, np.zeros((), j.dtype))

  return j

AXES_RULES[lax.dot_general_p] = _dot_general_a
JACOBIAN_RULES[lax.dot_general_p] = _dot_general_j


def _conv_general_dilated_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  lhs_spec, rhs_spec, out_spec = eqn.params['dimension_numbers']

  if idx == 1:
    in_trace = (rhs_spec[0],)
    out_trace = (out_spec[1],)
  else:
    raise NotImplementedError(eqn, idx)

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,))

def _conv_general_dilated_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  if (eqn.params['feature_group_count'] != 1 or
      eqn.params['batch_group_count'] != 1):
    raise NotImplementedError(eqn)

  lhs = invals[1 if idx == 0 else 0]
  rhs = invals[idx].aval
  ndim = cts_in.ndim

  lhs_spec, rhs_spec, out_spec = eqn.params['dimension_numbers']
  precision = eqn.params['precision']
  if isinstance(precision, tuple):
    if precision[0] == precision[1]:
      precision = precision[0]
    else:
      raise NotImplementedError(precision)

  j = lax.conv_general_dilated_patches(
      lhs=lhs,
      filter_shape=tuple(rhs.shape[i] for i in rhs_spec[2:]),
      window_strides=eqn.params['window_strides'],
      padding=eqn.params['padding'],
      lhs_dilation=eqn.params['lhs_dilation'],
      rhs_dilation=eqn.params['rhs_dilation'],
      dimension_numbers=eqn.params['dimension_numbers'],
      precision=precision,
      preferred_element_type=eqn.params['preferred_element_type']
  )

  j = np.moveaxis(j, out_spec[1], -1)
  j = np.expand_dims(j, out_spec[1])

  rhs_shape = tuple(1 if i == 0 else rhs.shape[s]
                    for i, s in enumerate(rhs_spec))
  j = j.reshape(j.shape[:ndim] + rhs_shape)
  source = tuple(range(ndim, j.ndim))
  target = tuple(ndim + r for r in rhs_spec)
  j = np.moveaxis(j, source, target)

  out_c = rhs.shape[rhs_spec[0]]
  eye = np.eye(out_c, dtype=np.bool_)
  eye = np.expand_dims(eye, [i
                             for i in range(j.ndim)
                             if i not in (out_spec[1], ndim + rhs_spec[0])])
  j = np.where(eye, j, np.zeros((), j.dtype))
  return j

AXES_RULES[lax.conv_general_dilated_p] = _conv_general_dilated_a
JACOBIAN_RULES[lax.conv_general_dilated_p] = _conv_general_dilated_j


def _add_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  inval = invals[idx]
  ndim = inval.ndim

  other = invals[1 if idx == 0 else 0]

  out_broadcast = ()
  in_broadcast = ()

  if other.ndim == 0:
    # Adding a scalar
    out_trace = tuple(range(ndim))

  else:
    # Adding a broadcastable array.
    out_trace = ()

    for i in range(ndim):
      if other.shape[i] in (inval.shape[i], 1):
        # Other array is broadcasted.
        out_trace += (i,)

      elif inval.shape[i] == 1:
        # This array is broadcasted
        out_broadcast += (i,)
        in_broadcast += ((i, None) if idx == 1 else (None, i),)

      else:
        raise ValueError(inval.shape, other.shape)

  in_trace = out_trace
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(0, 1),
              out_diagonal=(),
              in_diagonal=(),
              out_broadcast=out_broadcast,
              in_broadcast=in_broadcast)

def _add_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray,
    is_sub: bool
) -> np.ndarray:
  j = np.eye(utils.size_at(invals[idx]), dtype=invals[idx].aval.dtype)
  j = j.reshape(invals[idx].aval.shape * 2)
  j = np.broadcast_to(j, cts_in.shape + invals[idx].aval.shape)
  if is_sub and idx == 1:
    j = -j
  return j

AXES_RULES[lax.add_p] = _add_a
JACOBIAN_RULES[lax.add_p] = functools.partial(_add_j, is_sub=False)

AXES_RULES[ad.add_jaxvals_p] = _add_a
JACOBIAN_RULES[ad.add_jaxvals_p] = functools.partial(_add_j, is_sub=False)

AXES_RULES[lax.sub_p] = _add_a
JACOBIAN_RULES[lax.sub_p] = functools.partial(_add_j, is_sub=True)


def _mul_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  inval = invals[idx]
  ndim = inval.ndim
  other = invals[1 if idx == 0 else 0]

  out_diagonal = ()
  in_diagonal = ()

  if other.ndim == 0:
    # Multiplication by a scalar
    out_trace = tuple(range(ndim))

  else:
    # Multiplication by a broadcastable array.
    out_trace = ()
    for i in range(ndim):
      if other.shape[i] == 1:
        # Axis `i` is multiplied by a scalar.
        out_trace += (i,)

      else:

        if other.shape[i] == inval.shape[i]:
          out_diagonal += (i,)
          in_diagonal += ((i, i),)

        elif inval.shape[i] == 1:
          # This array is broadcasted
          pass

        else:
          raise ValueError(inval.shape, other.shape)

  in_trace = out_trace
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=out_diagonal,
              in_diagonal=in_diagonal,
              out_broadcast=(),
              in_broadcast=())

def _mul_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray,
    is_div: bool
) -> np.ndarray:
  if is_div and idx != 0:
    raise ValueError(eqn, idx)

  inval = invals[idx].aval
  if inval.size == 0:
    return np.zeros(cts_in.shape + inval.shape, inval.dtype)

  other = invals[1 if idx == 0 else 0]
  if is_div:
    other = np.ones((), other.dtype) / other

  if inval.ndim == 0:
    return other

  if other.ndim == 0:
    other = np.broadcast_to(other, inval.shape)

  assert other.ndim == inval.ndim == cts_in.ndim

  j = np.broadcast_to(other, cts_in.shape).reshape((-1,))
  j = np.diag(j)
  j = j.reshape(cts_in.shape * 2)

  sum_axes = ()
  for i in range(inval.ndim):
    if inval.shape[i] == 1:
      sum_axes += (cts_in.ndim + i,)

  j = np.sum(j, axis=sum_axes, keepdims=True)
  return j

AXES_RULES[lax.mul_p] = _mul_a
JACOBIAN_RULES[lax.mul_p] = functools.partial(_mul_j, is_div=False)

AXES_RULES[lax.div_p] = _mul_a
JACOBIAN_RULES[lax.div_p] = functools.partial(_mul_j, is_div=True)


# N-ARY PRIMITIVES


def _concatenate_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  dimension = eqn.params['dimension']

  out_trace = tuple(i for i in range(cts_in.ndim) if i != dimension)
  in_trace = out_trace

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=tuple(range(len(invals))))

def _concatenate_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  dimension = eqn.params['dimension']

  js = []
  inval = invals[idx].aval if hasattr(invals[idx], 'aval') else invals[idx]
  for i in range(len(invals)):
    inval_i = invals[i].aval if hasattr(invals[i], 'aval') else invals[i]
    inval_i_shape = tuple(inval_i.shape[k] if k == dimension else
                          inval.shape[k] for k in range(inval.ndim))

    if i == idx:
      j = np.eye(inval.size, dtype=inval.dtype)
    else:
      inval_i_size = onp.prod(inval_i_shape)
      j = np.zeros((inval_i_size, inval.size), inval.dtype)

    j = j.reshape(inval_i_shape + inval.shape)
    js.append(j)

  j = lax.concatenate(js, dimension)
  j = j.reshape(cts_in.shape + inval.shape)
  return j

AXES_RULES[lax.concatenate_p] = _concatenate_a
JACOBIAN_RULES[lax.concatenate_p] = _concatenate_j


# UNARY PRIMITIVES


def _rev_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  dimensions = eqn.params['dimensions']
  in_trace = out_trace = tuple(i for i in range(invals[idx].ndim)
                               if i not in dimensions)

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _rev_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx].aval
  j = _eye_like(cts_in, inval)
  j = lax.rev(j, eqn.params['dimensions'])
  return j

AXES_RULES[lax.rev_p] = _rev_a
JACOBIAN_RULES[lax.rev_p] = _rev_j


def _broadcast_in_dim_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  broadcast_dimensions = eqn.params['broadcast_dimensions']

  out_trace = broadcast_dimensions
  in_trace = tuple(range(invals[idx].ndim))

  out_broadcast = tuple(i for i in range(cts_in.ndim)
                        if i not in broadcast_dimensions)

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=(),
              out_broadcast=out_broadcast,
              in_broadcast=((None,),) * len(out_broadcast)
              )

def _broadcast_in_dim_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx].aval
  j = np.eye(inval.size, dtype=inval.dtype)
  j = j.reshape(inval.shape * 2)
  j = np.broadcast_to(j, cts_in.shape + inval.shape)
  return j

AXES_RULES[lax.broadcast_in_dim_p] = _broadcast_in_dim_a
JACOBIAN_RULES[lax.broadcast_in_dim_p] = _broadcast_in_dim_j


def _reduce_sum_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  axes = eqn.params['axes']

  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(i for i in range(invals[idx].ndim) if i not in axes)

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _reduce_sum_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  inval = invals[idx].aval
  j = np.eye(cts_in.size, dtype=inval.dtype)
  j = j.reshape(cts_in.shape * 2)
  j = np.expand_dims(j, tuple(a + cts_in.ndim for a in  eqn.params['axes']))
  j = np.broadcast_to(j, cts_in.shape + inval.shape)
  return j

AXES_RULES[lax.reduce_sum_p] = _reduce_sum_a
JACOBIAN_RULES[lax.reduce_sum_p] = _reduce_sum_j


def _reduce_window_sum_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  out_trace = []
  for i in range(cts_in.ndim):
    if (eqn.params['base_dilation'][i] == 1 and
        eqn.params['padding'][i] == (0, 0) and
        eqn.params['window_dilation'][i] == 1 and
        eqn.params['window_dimensions'][i] == 1 and
        eqn.params['window_strides'][i] == 1):
      out_trace.append(i)

  in_trace = out_trace
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,))

AXES_RULES[lax.reduce_window_sum_p] = _reduce_window_sum_a


def _pad_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  padding_config = eqn.params['padding_config']

  out_trace = tuple(i for i in range(cts_in.ndim)
                    if padding_config[i] == (0, 0, 0))
  in_trace = out_trace

  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _pad_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  padding_config = eqn.params['padding_config']

  inval = invals[idx].aval
  j = np.eye(inval.size, dtype=inval.dtype)
  j = j.reshape(inval.shape * 2)
  for _ in range(inval.ndim):
    padding_config += ((0, 0, 0),)

  j = lax.pad(j, np.zeros((), j.dtype), padding_config)
  return j

AXES_RULES[lax.pad_p] = _pad_a
JACOBIAN_RULES[lax.pad_p] = _pad_j


def _reshape_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  # out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(range(invals[idx].ndim))
  out_trace = in_trace
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _reshape_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(invals[idx].aval, invals[idx].aval)
  return j

AXES_RULES[lax.reshape_p] = _reshape_a
JACOBIAN_RULES[lax.reshape_p] = _reshape_j


def _eye_a(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  """Use this for elementwise-linear in `p` primitives `y(p, x)`.

  Precisely, require that `y(p, x)_k(i) = g(x)(p_i)` for some function `g(x)`
  and an index bijection `k: i -> j`.

  Note: multiplication doesn't satisfy this, since `y(p, x)_i = g(p_i, x_i)`.

  In this case the derivative matrix `dy/dp` is a constant-diagonal matrix, and
  all input-output axes can be collapsed.
  """
  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(range(invals[idx].ndim))
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _eye_j(
    eqn: Optional[JaxprEqn],
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx].aval)
  return j


# Identity
AXES_RULES[None] = _eye_a
JACOBIAN_RULES[None] = _eye_j


def _neg_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx].aval)
  return -j

AXES_RULES[lax.neg_p] = _eye_a
JACOBIAN_RULES[lax.neg_p] = _neg_j


def _zeros_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  return np.zeros(cts_in.shape + invals[idx].aval.shape, cts_in.dtype)

AXES_RULES[jax.ad.zeros_like_p] = _eye_a
JACOBIAN_RULES[jax.ad.zeros_like_p] = _zeros_j


def _transpose_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  out_trace = tuple(int(a) for a in eqn.params['permutation'])
  in_trace = tuple(range(invals[idx].ndim))
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

def _transpose_j(
    eqn: JaxprEqn,
    idx: int,
    invals: List[Union[np.ndarray, UndefinedPrimal]],
    cts_in: ShapedArray
) -> np.ndarray:
  j = _eye_like(cts_in, invals[idx].aval)
  inval = invals[idx].aval
  j = j.reshape(inval.shape * 2)

  inval_dims = tuple(i + cts_in.ndim for i in range(cts_in.ndim))
  j = lax.transpose(j, eqn.params['permutation'] + inval_dims)
  j = j.reshape(cts_in.shape + invals[idx].aval.shape)
  return j

AXES_RULES[lax.transpose_p] = _transpose_a
JACOBIAN_RULES[lax.transpose_p] = _transpose_j


def _squeeze_a(
    eqn: JaxprEqn,
    idx: int,
    invals: List[ShapedArray],
    cts_in: ShapedArray
) -> Axes:
  out_trace = tuple(range(cts_in.ndim))
  in_trace = tuple(i for i in range(invals[idx].ndim)
                   if i not in eqn.params['dimensions'])
  return Axes(out_trace=out_trace,
              in_trace=in_trace,
              in_trace_idxs=(idx,),
              out_diagonal=(),
              in_diagonal=())

AXES_RULES[lax.squeeze_p] = _squeeze_a
JACOBIAN_RULES[lax.squeeze_p] = _eye_j


AXES_RULES[xla.device_put_p] = _eye_a
JACOBIAN_RULES[xla.device_put_p] = _eye_j


AXES_RULES[lax.convert_element_type_p] = _eye_a
JACOBIAN_RULES[lax.convert_element_type_p] = _eye_j


AXES_RULES[lax.real_p] = _eye_a
AXES_RULES[lax.imag_p] = _eye_a
AXES_RULES[lax.complex_p] = _eye_a
AXES_RULES[lax.conj_p] = _eye_a