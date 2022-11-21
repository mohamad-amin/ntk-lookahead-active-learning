import random

import torch
import os
import re
import time
import numpy as np
import gc
from copy import deepcopy
import jax
import jaxlib
from jax.interpreters import xla
import jax.numpy as jnp
import scipy as sp
from jax import tree_util, vmap, pmap, lax
from jax.ops import index, index_update
from scipy.special import log_softmax
from src.utils import util, gpu_util, predict
from functools import partial, lru_cache
from collections import Counter
from dppy.finite_dpps import FiniteDPP
from neural_tangents.utils import utils as nt_utils
from src.utils.probability_utils import project_into_probability_simplex, project_into_probability_simplex_torch
from src.utils.empirical import jacobian_calculator
from src.utils.al_util import init_centers
from src.utils import torch_to_flax

__LOSS_CROSS_ENTROPY = jnp.asarray([1])
__LOSS_OTHER = jnp.asarray([2])
__UNCERTAINTY_RISK = jnp.asarray([11])
__UNCERTAINTY_OTHER = jnp.asarray([12])


def get_full_data(data, indices, noise=0.0):
    X, y = [], []
    for idx in indices:
        Xi, yi = data[idx]['inputs'], data[idx]['labels']
        X.append(Xi.unsqueeze(0))
        y.append(yi)
    X = torch.cat(X)
    if noise > 0.0:
        X += torch.normal(0., 1., X.size()) * noise
    y = torch.tensor(y)
    return X, y


def get_ntk_input_shape(data_config, num_input_channels, old=False):
    crop_size = data_config['transform']['crop_size']
    input_shape = (-1 if old else 1, crop_size, crop_size, num_input_channels)
    return input_shape


def compute_entropy_np(logits, y_train_onehot=None):
    if y_train_onehot is None:
        nt = logits.shape[0]
        train_entropy = 0.
    else:
        nt = logits.shape[0] - y_train_onehot.shape[0]
        train_log_probs = log_softmax(logits[nt:], axis=1)
        train_entropy = -1.0 * y_train_onehot * train_log_probs

    log_probs = log_softmax(logits[:nt], axis=1)
    probs = np.exp(log_probs)
    entropy = -np.sum(np.sum(probs * log_probs, axis=1)) + train_entropy
    return entropy


def compute_risk_np(logits, y_train_onehot=None):
    if y_train_onehot is None:
        nt = logits.shape[0]
        train_risk = 0.
    else:
        nt = logits.shape[0] - y_train_onehot.shape[0]
        train_log_probs = log_softmax(logits[nt:], axis=1)
        train_probs = np.exp(train_log_probs)
        train_risk = np.sum(y_train_onehot - y_train_onehot * train_probs)

    log_probs = log_softmax(logits[:nt], axis=1)
    probs = np.exp(log_probs)
    max_indices = np.argmax(probs, axis=1)
    risk = np.sum(1 - probs[np.arange(nt), max_indices]) + train_risk
    return risk


def compute_risk_jnp(logits, y_train_onehot=None):
    if y_train_onehot is None:
        nt = logits.shape[0]
        train_risk = 0.
    else:
        nt = logits.shape[0] - y_train_onehot.shape[0]
        train_log_probs = jax.nn.log_softmax(logits[nt:], axis=1)
        train_probs = jnp.exp(train_log_probs)
        train_risk = jnp.sum(y_train_onehot - y_train_onehot * train_probs)

    log_probs = jax.nn.log_softmax(logits[:nt], axis=1)
    probs = jnp.exp(log_probs)
    max_indices = jnp.argmax(probs, axis=1)
    risk = jnp.sum(1 - probs[jnp.arange(nt), max_indices]) + train_risk
    return risk


def compute_l2_risk_np(probs, y_train_onehot=None):
    if y_train_onehot is None:
        nt = probs.shape[0]
        train_risk = 0.
    else:
        nt = probs.shape[0] - y_train_onehot.shape[0]
        train_probs = probs[nt:]
        train_risk = jnp.sum(y_train_onehot - y_train_onehot * train_probs)

    ones = jnp.ones_like(probs[:nt])
    indices = jnp.argmin(jnp.abs(ones - probs[:nt]), axis=1)
    unbounded_probs = probs[jnp.arange(nt), indices]
    bounded_probs = jnp.where(unbounded_probs >= 1., jnp.ones_like(unbounded_probs), unbounded_probs)
    risk = jnp.sum(1 - bounded_probs) + train_risk
    return risk


def compute_inf_l2_risk_jnp(probs): # nt x k
    probs = project_into_probability_simplex(probs, use_gpu=False)
    nt = probs.shape[0]
    indices = jnp.argmax(probs, axis=1)
    probs = probs[jnp.arange(nt), indices]
    risk = jnp.sum(1 - probs)  # todo: use kurtosis
    return risk


def compute_inf_entropy_jnp(probs):  # nt x k
    probs = project_into_probability_simplex(probs, use_gpu=False)
    pseudo_label_entropy = -jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-12), axis=1))
    return pseudo_label_entropy


def compute_entropy_jnp(logits): # k
    log_probs = jax.nn.log_softmax(logits, axis=1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=1))
    return entropy


def update_ntk_params(params, model, bn_with_running_stats):
    if bn_with_running_stats:
        return update_ntk_params_w_running_stats(params, model)
    else:
        return update_ntk_params_wo_running_stats(params, model)


def update_ntk_params_wo_running_stats(params, model):
    treedef = tree_util.tree_structure(params)
    new_jax_params = tree_util.tree_leaves(params)
    if hasattr(model, 'module'):
        model = model.module

    model.eval()
    for i, (name, param) in enumerate(model.named_parameters()):
        if i == len(new_jax_params) - 1:  # last fc layer bias
            new_param = param.view(1, -1).detach().cpu().numpy()
        elif i == len(new_jax_params) - 2:  # last fc layer weight
            new_param = param.detach().cpu().numpy().swapaxes(1, 0)
        elif re.findall("bn\d.weight", name):  # batch norm weight
            new_param = torch.diag(param).detach().cpu().numpy()
        elif re.findall("bn\d.bias", name):
            new_param = param.view(1, 1, 1, -1).detach().cpu().numpy()
        else:
            if len(param.shape) == 1:
                param = param.view(-1, 1, 1, 1)
            new_param = param.detach().cpu().numpy().transpose(2, 3, 1, 0)
        try:
            new_jax_params[i] = index_update(new_jax_params[i], index[:], new_param)
        except:
            import IPython; IPython.embed()

    new_jax_params = tree_util.tree_unflatten(treedef, new_jax_params)
    return new_jax_params


def update_ntk_params_w_running_stats(params, model):
    dtype = 'float64'
    return torch_to_flax.transfer_params_from_torch_model(model, params, dtype)


# Todo: this can be optimized by removing the for loop
def batch_apply_fn(apply_fn, params, rng, X, batch_size=1000):
    fx_0 = None
    for i in range(0, X.shape[0], batch_size):
        end_index = min(X.shape[0], i + batch_size)
        X_subset = X[i:end_index]
        if fx_0 is not None:
            fx_0 = jnp.concatenate(
                (fx_0, apply_fn(params, X_subset)), axis=0)
        else:
            fx_0 = apply_fn(params, X_subset)
    return fx_0


def calculate_maximum_ntk_batch_fitting_in_memory(num_classes, params_size, num_devices, coef, trace_axes=()):
    float_size = gpu_util.get_float_size()
    if torch.cuda.device_count() > 0:
        usable_memory = gpu_util.get_usable_gpu_memory_in_bytes()
    else:
        usable_memory = gpu_util.get_usable_cpu_memory_in_bytes()
    space_unit = (usable_memory * num_devices) / float_size
    effective_num_classes = num_classes if len(trace_axes) == 0 else 1
    return np.floor(space_unit / (effective_num_classes * params_size * coef))


def ntk_fn_dynamic_batched(ntk_fn_builder, num_devices, params, num_classes, params_size, x1, x2=None, c=12, trace_axes=()):
    x2a = x1 if x2 is None else x2
    n1, n2 = x1.shape[0], x2a.shape[0]
    maximum_batch_size = calculate_maximum_ntk_batch_fitting_in_memory(num_classes, params_size, num_devices, c, trace_axes)
    assert n1 * n2 % (num_devices * num_devices) == 0, "both {} and {} should be divisible by #GPUs!".format(n1, n2)
    optimal_batch_size = np.gcd(n1, n2) // num_devices
    divisors = util.get_sorted_divisors(optimal_batch_size)[::-1]
    try:
        batch_size = divisors[divisors < maximum_batch_size][0]
    except:
        batch_size = divisors[divisors <= 500][0]
    print('n1 - {} n2 - {} batch size - {}'.format(n1, n2, batch_size))
    return ntk_fn_builder(batch_size=batch_size)(x1, x2, params)


def pad_kernels_to_same_shape(k_test_train, k_train_train, fx_train_0, y_train_onehot, n, nt, num_classes, padded_size):
    k, p = num_classes, padded_size - 1  # the last one is to be added in the computations
    # n x n x k x k --> n x p x k x k
    k_train_train = np.concatenate((k_train_train, np.zeros((n, p - n, k, k))), axis=1)
    # create (p-n) x p x k x k
    train_kernel_second_block_row = np.concatenate((
        np.zeros((p - n, n, k, k)), np.eye((p - n) * k).reshape(p - n, k, p - n, k).transpose(0, 2, 1, 3)), axis=1)
    # train_train --> p x p x k x k
    k_train_train = np.concatenate((k_train_train, train_kernel_second_block_row), axis=0)
    # nt x n x k x k --> nt x p x k x k
    k_test_train = np.concatenate((k_test_train, np.zeros((nt, p - n, k, k))), axis=1)
    # n x k --> p x k
    fx_train_0 = np.concatenate((fx_train_0, np.zeros((p - n, k))), axis=0)
    y_train_onehot = np.concatenate((y_train_onehot, np.zeros((p - n, k))), axis=0)
    return k_test_train, k_train_train, fx_train_0, y_train_onehot


def transpose_wrt_trace_axes(mat, transpose_args, trace_axes=()):
    if len(trace_axes) == 0:
        return mat.transpose(*transpose_args)
    transpose_args = list(filter(lambda x: x in [0, 1], transpose_args))
    return mat.transpose(*transpose_args)


def get_uncertainty_ntk_block_sequential(dataloaders, models, subset_predictions, model_config, al_params,
                                         cycle, cycle_count, num_devices=1):

    labeled_dataset, unlabeled_dataset = dataloaders['train'].dataset, dataloaders['unlabeled'].dataset
    labeled_set, subset = dataloaders['labeled_set'], dataloaders['subset']

    apply_fn, loss_fn, ntk_fn_batched, params, rng = \
        models['apply_fn'], models['loss_fn'], models['ntk_fn'], models['ntk_params'], models['rng']

    num_classes, ntk_config = \
        model_config['model_arch']['num_classes'], model_config['ntk']

    trace_axes = ntk_config.get('trace_axes', [])

    # Using diag_reg to prevent numerical problems...
    diag_reg = ntk_config.get('ntk_diag_reg', 1e-5)
    diag_reg_per_cycle = ntk_config.get('ntk_diag_reg_pc', 1e-5)
    diag_reg_starting_cycle = ntk_config.get('ntk_diag_reg_sc', 5)
    if cycle < diag_reg_starting_cycle:
        diag_reg = 0.0
    else:
        diag_reg = diag_reg + diag_reg_per_cycle * (cycle - diag_reg_starting_cycle)

    batch_size, t, learning_rate, momentum = \
        ntk_config['ntk_batch'], ntk_config['t'], ntk_config['lr'], ntk_config['momentum']
    if isinstance(t, str):
        t = eval(t)

    eps_end = ntk_config.get('eps_end', 0.0)
    eps_method = ntk_config.get('eps_method', 'poly')
    if eps_method == 'linear':
        eps = cycle / cycle_count
    else:
        eps = (cycle / cycle_count) ** 1.5
    eps *= eps_end

    ntk_objective = ntk_config.get('ntk_objective', 'pseudo_contrastive')
    use_dpp = ntk_config.get('dpp', False)
    ntk_fn_builder = models['ntk_fn_builder']
    ntk_trace_fn_builder = models['ntk_trace_fn_builder']
    dynamic_ntk_batch_coef = ntk_config.get('kernel_comp_coef', 12)

    dynamic_batched_ntk_fn = partial(
        ntk_fn_dynamic_batched, ntk_fn_builder,
        num_devices, params, num_classes, models['ntk_params_size'], c=dynamic_ntk_batch_coef, trace_axes=trace_axes)

    dynamic_batched_ntk_trace_fn = partial(ntk_fn_dynamic_batched, ntk_trace_fn_builder,
                                     num_devices, params, num_classes, models['ntk_params_size'], c=6.5)

    smart_transpose = partial(transpose_wrt_trace_axes, trace_axes=trace_axes)

    print('NTK Objective:', ntk_objective)

    print('Intersection of labeled and unlabeled set:')
    print(np.intersect1d(labeled_set, subset))

    ntk_data_noise = ntk_config.get('data_noise', 0.)

    X_train, y_train = get_full_data(labeled_dataset, labeled_set, noise=ntk_data_noise)
    X_test, y_test = get_full_data(unlabeled_dataset, subset, noise=ntk_data_noise)

    count_train = Counter(y_train.numpy())
    ordered_count = sorted(count_train.items(), key=lambda i: i[0])
    print('y train: {}'.format(str(ordered_count)), flush=True)

    count_test = Counter(y_test.numpy())
    ordered_count = sorted(count_test.items(), key=lambda i: i[0])
    print('y test: {}'.format(str(ordered_count)), flush=True)

    X_train = jnp.asarray(X_train.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=jnp.float64)
    y_train_onehot = np.zeros((len(y_train), num_classes))
    y_train_onehot[np.arange(len(y_train)), y_train] = 1
    y_train_onehot = jnp.array(y_train_onehot, dtype=jnp.float64)

    X_test = jnp.asarray(X_test.detach().cpu().numpy().transpose(0, 2, 3, 1), dtype=jnp.float64)
    y_onehots = jnp.eye(num_classes, dtype=jnp.float64)
    y_test_onehot = np.zeros((len(y_test), num_classes))
    y_test_onehot[np.arange(len(y_test)), y_test] = 1
    y_test_onehot = jnp.array(y_test_onehot, dtype=jnp.float64)
    print('X train shape: {}'.format(X_train.shape), flush=True)
    print('X test shape: {}'.format(X_test.shape), flush=True)


    # compute the common kernel of the full training
    n = X_train.shape[0]
    nt = X_test.shape[0]
    k = num_classes

    # compute initial predictions
    prev_time = time.time()
    fx_train_0 = batch_apply_fn(
        apply_fn, params, rng, X_train, batch_size=100)
    fx_test_0 = batch_apply_fn(
        apply_fn, params, rng, X_test, batch_size=100)
    print('initial prediction is done: {}'.format(time.time() - prev_time), flush=True)

    print('Jax Train Accuracy:', (fx_train_0.argmax(axis=1) == y_train_onehot.argmax(axis=1)).mean())
    print('Jax Test Accuracy:', (fx_test_0.argmax(axis=1) == y_test_onehot.argmax(axis=1)).mean())
    print('Torch Test Accuracy:', (subset_predictions.argmax(axis=1) == y_test_onehot.argmax(axis=1)).mean())

    print('Available CPU RAM before kernel computations: {:.2f} GB'.format(
        gpu_util.get_usable_cpu_memory_in_bytes() / (2**30)))


    # k_train_train = np.eye((n * num_classes)).reshape(n, num_classes, n, num_classes).transpose(0, 2, 1, 3)
    prev_time = time.time()
    k_train_train = np.array(dynamic_batched_ntk_fn(X_train, None))  # n x n x kn x kn
    print('train train is done: {}'.format(time.time() - prev_time), flush=True)
    prev_time = time.time()
    k_test_train = np.array(dynamic_batched_ntk_fn(X_test, X_train))  # nt x n x knt x kn
    print('test train is done: {}'.format(time.time() - prev_time), flush=True)
    prev_time = time.time()
    k_test_test = np.array(dynamic_batched_ntk_fn(X_test, None))  # nt x nt x k x k
    print('test test done: {}'.format(time.time() - prev_time), flush=True)

    if diag_reg > 0.0:  # Todo: make this compatible with trace_axes
        print(f'Regularizing k_train_train with {diag_reg}')
        prev_time = time.time()
        dimension = n * k
        batch = 20000
        A = np.array(k_train_train.transpose(0, 2, 1, 3).reshape((n * k, n * k)))
        del k_train_train
        d_diag_reg = diag_reg * np.trace(A) / dimension
        for i in range(0, n * k, batch):  # So that we don't create a new nk * nk matrix (identity)
            s = i
            e = min(i + batch, n * k)
            A[s:e, s:e] += d_diag_reg * np.eye(e - s)
        k_train_train = A.reshape(n, k, n, k).transpose(0, 2, 1, 3)
        del A
        print('Done with regularization in {} sec'.format(time.time() - prev_time))
    else:
        print('No diagonal regularization')

    backend = jax.lib.xla_bridge.get_backend()
    live_buffers_at_start = backend.live_buffers()
    live_executables_at_start = backend.live_executables()
    print('Live buffers count:', len(live_buffers_at_start))
    print('Live executables count:', len(live_executables_at_start))

    @partial(jax.jit, backend='cpu')
    def predict_normal_mse(k_train_train, y_train_onehot, fx_train_0, fx_test_0, k_test_train, learning_rate):
        return predict.gradient_descent_mse(
            k_train_train, y_train_onehot, learning_rate, trace_axes=tuple(trace_axes), diag_reg=0.
        )(None, fx_train_0, fx_test_0, k_test_train)[1]

    if ntk_config.get('report_ntk_accuracy', False):
        ntk_fx_test_inf = predict_normal_mse(k_train_train, y_train_onehot, fx_train_0, fx_test_0, k_test_train, None)
        print('NTK Test Accuracy:', (ntk_fx_test_inf.argmax(axis=1) == y_test_onehot.argmax(axis=1)).mean())

    if ntk_config.get('use_true_uncertainties', False):
        induced_probs = project_into_probability_simplex_torch(y_test_onehot)
    else:
        induced_probs = project_into_probability_simplex_torch(fx_test_0)

    # Conversion to numpy because it's way faster to work with outside of jax!
    k_train_train, k_test_train, k_test_test = np.array(k_train_train), np.array(k_test_train), np.array(k_test_test)
    fx_train_0, fx_test_0 = np.array(fx_train_0), np.array(fx_test_0)
    y_onehots, y_train_onehot = np.array(y_onehots), np.array(y_train_onehot)
    induced_probs = np.array(induced_probs)

    def select_top_point_according_to_objective(fx_test_ts, ntk_fx_test_0, y_test, selections):

        sigmoid_temperature = ntk_config.get('sigmoid_temperature', 4.0)

        k = fx_test_ts.shape[1]
        n_y_test = y_test.detach().cpu().numpy()

        def compute_accuracy(fx, y):
            return (fx.argmax(axis=1) == y).mean()

        def compute_temperature_norm(fx, y):
            return jnp.linalg.norm(jax.nn.sigmoid(fx / sigmoid_temperature) - jax.nn.sigmoid(y / sigmoid_temperature))

        compute_accuracies = lambda fxs, y: vmap(compute_accuracy, in_axes=(0, None))(fxs, y)
        compute_temp_norms = lambda fxs, y: vmap(compute_temperature_norm, in_axes=(0, None))(fxs, y)
        compute_risks = lambda fxs: vmap(vmap(compute_inf_l2_risk_jnp))(fxs)

        compute_batch_accuracies = lambda fx_tests, y: \
            vmap(lambda fxs, y: compute_accuracies(fxs, y), in_axes=(0, None))(fx_tests, y)

        p_compute_risk = pmap(compute_risks)
        p_compute_acc = pmap(compute_accuracies, in_axes=(0, None))
        p_compute_temp_norm = pmap(compute_temp_norms, in_axes=(0, None))

        p_compute_batch_acc = pmap(compute_batch_accuracies, in_axes=(0, None))

        if ntk_objective == 'pseudo_contrastive':
            n_accuracies_with_pseudo_labels = p_compute_acc(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0.argmax(axis=1)
            ).reshape(-1,)
            sorted_indices = np.argsort(n_accuracies_with_pseudo_labels)
        elif ntk_objective == 'expected_contrastive':
            n_expected_accuracies = jnp.einsum('ik,ik->i', p_compute_batch_acc(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0.argmax(axis=1)
            ).reshape(-1, k), induced_probs)
            sorted_indices = np.argsort(n_expected_accuracies)
        elif ntk_objective == 't_softmax':
            n_softmax_difference_with_pseudo_labels = p_compute_temp_norm(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0
            ).reshape(-1, )
            sorted_indices = np.argsort(n_softmax_difference_with_pseudo_labels)[::-1]
        elif ntk_objective == 'pseudo_er':
            n_er_with_pseudo_labels = p_compute_risk(
                gpu_util.gpu_split(fx_test_ts)
            ).reshape(-1, )
            sorted_indices = np.argsort(n_er_with_pseudo_labels)[::-1]

        actual_acc_point = next(i for i in sorted_indices if i not in selections)
        return actual_acc_point

    def select_batch_according_to_objective(fx_test_ts, ntk_fx_test_0, y_test, add_num):

        T = ntk_config.get('softmax_temperature', 1.0)
        print(f'Temperature is: {T}')

        k = fx_test_ts.shape[1]
        n_y_test = y_test.detach().cpu().numpy()

        def compute_accuracy(fx, y):
            return (fx.argmax(axis=1) == y).mean()

        def compute_norm_diff(fx, y):
            return jnp.linalg.norm(fx - y)

        def soft_log_norm(probs):
            return jax.nn.softmax(jnp.log(project_into_probability_simplex(probs, use_gpu=False)) / T)

        def compute_temperature_norm(fx, y):
            return jnp.linalg.norm(soft_log_norm(fx) - soft_log_norm(y))

        compute_accuracies = lambda fxs, y: vmap(compute_accuracy, in_axes=(0, None))(fxs, y)
        compute_norm_diffs = lambda fxs, y: vmap(compute_norm_diff, in_axes=(0, None))(fxs, y)
        compute_temp_norms = lambda fxs, y: vmap(compute_temperature_norm, in_axes=(0, None))(fxs, y)
        compute_risks = lambda fxs: vmap(vmap(compute_inf_l2_risk_jnp))(fxs)

        compute_batch_accuracies = lambda fx_tests, y: \
            vmap(lambda fxs, y: compute_accuracies(fxs, y), in_axes=(0, None))(fx_tests, y)
        compute_batch_norm_diffs = lambda fx_tests, y: \
            vmap(lambda fxs, y: compute_norm_diffs(fxs, y), in_axes=(0, None))(fx_tests, y)

        p_compute_risk = pmap(compute_risks)
        p_compute_acc = pmap(compute_accuracies, in_axes=(0, None))
        p_compute_norm_diff = pmap(compute_norm_diffs, in_axes=(0, None))
        p_compute_temp_norm = pmap(compute_temp_norms, in_axes=(0, None))

        p_compute_batch_acc = pmap(compute_batch_accuracies, in_axes=(0, None))
        p_compute_batch_diff = pmap(compute_batch_norm_diffs, in_axes=(0, None))

        if ntk_objective == 'pseudo_contrastive':
            n_accuracies_with_pseudo_labels = p_compute_acc(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0.argmax(axis=1)
            ).reshape(-1,)
            sorted_indices = np.argsort(n_accuracies_with_pseudo_labels)
        elif ntk_objective == 'expected_contrastive':
            n_expected_accuracies = jnp.einsum('ik,ik->i', p_compute_batch_acc(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0.argmax(axis=1)
            ).reshape(-1, k), induced_probs)
            sorted_indices = np.argsort(n_expected_accuracies)
        elif ntk_objective == 't_softmax':
            n_softmax_difference_with_pseudo_labels = p_compute_temp_norm(
                gpu_util.gpu_split(fx_test_ts), ntk_fx_test_0
            ).reshape(-1, )
            sorted_indices = np.argsort(n_softmax_difference_with_pseudo_labels)[::-1]
        elif ntk_objective == 'pseudo_er':
            n_er_with_pseudo_labels = p_compute_risk(
                gpu_util.gpu_split(np.expand_dims(fx_test_ts, 1))
            ).reshape(-1, )
            sorted_indices = np.argsort(n_er_with_pseudo_labels)
        elif ntk_objective == 'eer':
            n_eer = jnp.einsum('ik,ik->i', p_compute_risk(
                gpu_util.gpu_split(fx_test_ts)
            ).reshape(-1, k), induced_probs)
            sorted_indices = np.argsort(n_eer)

        return np.array(sorted_indices)

    def compute_class_uncertainty(extra_y_train_onehot, extra_k_test_train_col, extra_k_train_train_col,
                                  extra_k_train_train_point, extra_fx_train_0, y_train_onehot,
                                  staged_x_non_channel_shape,
                                  staged_A, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first,
                                  k_test_train):
        fx_test_t = predict.block_gradient_descent_mse_staged(
            k_test_train, staged_x_non_channel_shape, staged_A, staged_C, staged_rhs,
            staged_orig_preds, staged_odd, staged_first, extra_k_test_train_col,
            extra_k_train_train_col, extra_k_train_train_point, extra_y_train_onehot, extra_fx_train_0)
        return 0, fx_test_t  # 1, nt x k

    def compute_batch_uncertainty(pseudo_label, induced_probs, extra_k_train_train_col, extra_k_train_train_point,
                                  extra_k_test_train_col,
                                  extra_fx_train_0, y_onehots, y_train_onehot, staged_x_non_channel_shape,
                                  staged_A, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first,
                                  k_test_train):
        extra_k_train_train_col = smart_transpose(extra_k_train_train_col, (1, 0, 2, 3))  # for vectorizing over axis 0 (n x 1 x k x k)
        extra_k_test_train_col = smart_transpose(extra_k_test_train_col, (1, 0, 2, 3))  # for vectorizing over axis 0 (nt x 1 x k x k)

        y_onehots = y_onehots[extra_fx_train_0.argmax(axis=1)]
        y_onehots = y_onehots[pseudo_label]
        _, fx_test_t = compute_class_uncertainty(
            y_onehots, extra_k_test_train_col, extra_k_train_train_col,
            extra_k_train_train_point, extra_fx_train_0, y_train_onehot, staged_x_non_channel_shape,
            staged_A, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train)

        return 0, fx_test_t

    v_compute_batch_uncertainty = vmap(compute_batch_uncertainty, in_axes=tuple([0] * 6 + [None] * 10))

    @partial(jax.jit, backend='cpu', static_argnums=(8, 9, 13, 14))
    def jv_compute_batch_uncertainty(batched_pseudo_labels, batched_induced_probs,
                                     extra_k_train_train_col_batch, extra_k_train_train_point_batch,
                                     extra_k_test_train_col_batch, extra_fx_train_0_batch, y_onehots, y_train_onehot,
                                     staged_x_non_channel_shape, staged_A, staged_C,
                                     staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train):

        batched_extra_k_train_train_point_batch = jnp.stack([
            extra_k_train_train_point_batch[i:i + 1, i:i + 1] for i in
            range(extra_k_train_train_point_batch.shape[0])]).squeeze(1)
        extra_k_train_train_col_batch = smart_transpose(extra_k_train_train_col_batch, (1, 0, 2, 3))  # for vectorizing over axis 0
        extra_k_test_train_col_batch = smart_transpose(extra_k_test_train_col_batch, (1, 0, 2, 3))  # for vectorizing over axis 0
        return v_compute_batch_uncertainty(
            batched_pseudo_labels, batched_induced_probs, jnp.expand_dims(extra_k_train_train_col_batch, 1),
            jnp.expand_dims(batched_extra_k_train_train_point_batch, 1),
            jnp.expand_dims(extra_k_test_train_col_batch, 1),
            jnp.expand_dims(extra_fx_train_0_batch, 1), y_onehots, y_train_onehot, staged_x_non_channel_shape,
            staged_A, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train
        )

    def prepare_data_for_p_compute(batched_pseudo_labels, batched_induced_probs,
                                   extra_k_train_train_col_batch, extra_k_train_train_point_batch,
                                   extra_k_test_train_col_batch, extra_fx_train_0_batch, y_onehots, y_train_onehot,
                                   staged_x_non_channel_shape, staged_A, staged_C,
                                   staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train, use_gpu):

        batched_extra_k_train_train_point_batch = np.stack([
            extra_k_train_train_point_batch[i:i + 1, i:i + 1] for i in
            range(extra_k_train_train_point_batch.shape[0])]).squeeze(1)
        gs = partial(gpu_util.gpu_split, use_gpu=use_gpu)
        return (gs(batched_pseudo_labels), gs(batched_induced_probs),
                gs(np.expand_dims(extra_k_train_train_col_batch.transpose(1, 0, 2, 3), 1)),
                gs(np.expand_dims(batched_extra_k_train_train_point_batch, 1)),
                gs(np.expand_dims(extra_k_test_train_col_batch.transpose(1, 0, 2, 3), 1)),
                gs(np.expand_dims(extra_fx_train_0_batch, 1)), y_onehots, y_train_onehot, staged_x_non_channel_shape,
                staged_A, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train)

    def p_compute_batch_uncertainty(pfunc, dfunc, batched_pseudo_labels, batched_induced_probs,
                                    extra_k_train_train_row_batch, extra_k_train_train_point_batch,
                                    extra_k_test_train_col_batch, extra_fx_train_0_batch, y_onehots, y_train_onehot,
                                    staged_x_non_channel_shape, staged_A, staged_C,
                                    staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train, use_gpu):
        data = list(dfunc(
            batched_pseudo_labels, batched_induced_probs,
            extra_k_train_train_row_batch, extra_k_train_train_point_batch,
            extra_k_test_train_col_batch, extra_fx_train_0_batch, y_onehots, y_train_onehot,
            staged_x_non_channel_shape, staged_A, staged_C,
            staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train, use_gpu))
        uncrts, fx_test_t = pfunc(*data)
        return uncrts.reshape(-1, *uncrts.shape[2:]), fx_test_t.reshape(-1, *fx_test_t.shape[2:])

    def compute_dpp_similarity(expected_fx_test_t, expected_fx_test_ts_transposed):
        sim_subset = jnp.exp(-jnp.sqrt(jnp.sum(
            (jnp.expand_dims(expected_fx_test_t, 0) - expected_fx_test_ts_transposed) ** 2, axis=(-2, -1))))
        return sim_subset.squeeze(0)

    @lru_cache(1)
    def construct_p_compute_dpp_similarity():
        return pmap(vmap(compute_dpp_similarity, in_axes=(0, None)), backend='gpu')

    def p_compute_similarity(expected_fx_test_ts_replicated, expected_fx_test_ts_transposed_replicated):
        sim_subset = construct_p_compute_dpp_similarity()(
            gpu_util.gpu_split(expected_fx_test_ts_replicated), expected_fx_test_ts_transposed_replicated
        )
        return sim_subset.reshape(-1, *sim_subset.shape[2:])

    @lru_cache(1)
    def compute_dpp_batch_size():
        float_size = gpu_util.get_float_size()
        usable_gpu_memory = gpu_util.get_usable_gpu_memory_in_bytes()
        unit = nt * nt * num_classes

        def step_multiplier_fits(sb):
            approx = .9
            return usable_gpu_memory * num_devices * approx > float_size * unit * (3 * sb)

        max_step_size = 1
        while step_multiplier_fits(max_step_size * num_devices):
            max_step_size += 1
        max_step_size -= 1
        divisors = util.get_sorted_divisors(nt // num_devices)[::-1]
        step_size = divisors[divisors < max_step_size][0]
        return step_size * num_devices

    def compute_dpp_sim_matrix(fx_test_ts, induced_probs, nt, b_size, expose=False):

        induced_probs = induced_probs.reshape(induced_probs.shape + (1, 1))
        fx_test_ts = np.sum(fx_test_ts * induced_probs, axis=1, keepdims=True)
        fx_test_ts_transposed = np.transpose(fx_test_ts, (1, 0, 2, 3))

        fx_test_ts_transposed_replicated = jax.device_put_replicated(fx_test_ts_transposed, jax.devices())

        sim_matrix = []
        for idx in range(0, nt, b_size):
            sim_subset = p_compute_similarity(
                fx_test_ts[idx:idx + b_size], fx_test_ts_transposed_replicated)
            sim_matrix.append(sim_subset)
        sim_matrix = jnp.concatenate(sim_matrix, axis=0)

        return sim_matrix

    def get_sorted_points_with_dpp(expected_uncertainties, sim_matrix, nt, gamma, add_num):
        expected_uncertainties = expected_uncertainties.reshape(-1, 1)
        expected_uncertainties = nt + expected_uncertainties
        kernel = np.power(expected_uncertainties, gamma) * sim_matrix * np.power(expected_uncertainties, gamma).T
        DPP = FiniteDPP('likelihood', **{'L': kernel})
        dpp_idx = DPP.sample_exact_k_dpp(size=add_num)
        sorted_points = np.arange(kernel.shape[0])[dpp_idx]
        return sorted_points

    def compute_uncertainties(nt, n, k_train_train, k_test_train, k_test_test, induced_probs,
                              fx_train_0, fx_test_0, y_onehots, y_train_onehot, num_classes, learning_rate, num_devices,
                              p_compute_batch_uncertainty, jv_compute_batch_uncertainty,
                              parallel_batch_computation_func,
                              prepare_data_for_p_compute, pad=False, p=-1, expose=True, use_loop=False, diag_reg=0.,
                              dpp=False, selections=None, pseudo_labels=None):

        p = n if not pad else p
        effective_num_classes = num_classes if len(trace_axes) == 0 else 1
        float_size = gpu_util.get_float_size()
        usable_gpu_memory = gpu_util.get_usable_gpu_memory_in_bytes()
        usable_cpu_memory = gpu_util.get_usable_cpu_memory_in_bytes()

        def step_multiplier_fits(sb, gpu=True):
            approx = .8
            if gpu:
                return usable_gpu_memory * num_devices * approx > (
                        float_size * (effective_num_classes ** 2) * (2 * p + sb) * (nt + p)) and \
                       usable_gpu_memory * approx > (float_size * (effective_num_classes ** 2) * 2 * p * (nt + p))
            else:
                return usable_cpu_memory * approx > (
                        float_size * (effective_num_classes ** 2) * 2 * (nt + sb) * (nt + p))

        step_multiplier = 0
        use_gpu = False  # FIXME
        num_devices = num_devices if use_gpu else jax.device_count('cpu')
        if num_devices >= nt:
            step_multiplier = 1
        else:
            while step_multiplier_fits(num_devices * step_multiplier, gpu=use_gpu):
                step_multiplier += 1
            step_multiplier -= 1
        step_size = num_devices * step_multiplier

        print('Using GPU? {} - Step size: {} - num devices: {}'.format(use_gpu, step_size, num_devices))

        if pad:
            k_test_train, k_train_train, fx_train_0, y_train_onehot = \
                pad_kernels_to_same_shape(k_test_train, k_train_train, fx_train_0, y_train_onehot, n, nt, num_classes, p)

        prev_time = time.time()

        staged_x_non_channel_shape, staged_C, staged_rhs, staged_orig_preds, staged_odd, staged_first = \
            predict.prepare_gradient_descent_mse_staged(
                k_train_train, k_test_train, y_train_onehot, fx_train_0, fx_test_0, learning_rate, diag_reg,
                trace_axes=tuple(trace_axes))

        staged_A = k_train_train.shape[1] * effective_num_classes
        original_fx_test_t = staged_orig_preds

        if pseudo_labels is None:
            pseudo_labels = original_fx_test_t.argmax(axis=1)

        if expose:
            print('Preparing predict took {}sec'.format(time.time() - prev_time))

        fx_test_ts = []
        expected_uncertainties = []
        for i in range(0, nt, step_size):  # sorry, this is because of memory issues

            step_batch_size = min(step_size, nt - i)
            if i % 100 == 0 and expose:
                prev_time = time.time()
                print('Test batch size for uncertainty computation: {}, n: {}, p: {}'.format(
                    step_batch_size, n, p), flush=True)

            # Note: if padded, n would be p in the comments
            extra_k_train_train_col_batch_ = smart_transpose(k_test_train[i:i + step_batch_size], (1, 0, 3, 2)) # n x step_batch x k x k
            extra_k_train_train_point_batch = k_test_test[i:i + step_batch_size, i:i + step_batch_size]  # step_batch x step_batch x k x k
            extra_k_test_train_col_batch = k_test_test[:, i:i + step_batch_size]  # nt x step_batch x k x k
            extra_fx_train_0_batch = fx_test_0[i:i + step_batch_size]  # step_batch x num_classes
            batched_induced_probs = induced_probs[i:i + step_batch_size]
            batched_pseudo_labels = pseudo_labels[i:i + step_batch_size]

            if use_loop:
                raise NotImplementedError('Todo: Implement if needed')
            else:
                if jax.device_count() == 0 or not use_gpu:
                    batch_uncertainties, fx_test_t = jv_compute_batch_uncertainty(
                        batched_pseudo_labels, batched_induced_probs, extra_k_train_train_col_batch_,
                        extra_k_train_train_point_batch,
                        extra_k_test_train_col_batch, extra_fx_train_0_batch, y_onehots, y_train_onehot,
                        staged_x_non_channel_shape, staged_A, staged_C,
                        staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train)
                else:
                    batch_uncertainties, fx_test_t = p_compute_batch_uncertainty(
                        parallel_batch_computation_func, prepare_data_for_p_compute,
                        batched_pseudo_labels, batched_induced_probs,
                        extra_k_train_train_col_batch_, extra_k_train_train_point_batch, extra_k_test_train_col_batch,
                        extra_fx_train_0_batch, y_onehots, y_train_onehot,
                        staged_x_non_channel_shape, staged_A, staged_C,
                        staged_rhs, staged_orig_preds, staged_odd, staged_first, k_test_train, use_gpu)

            fx_test_ts.append(fx_test_t)
            expected_uncertainties.extend(batch_uncertainties)
            if i % 100 == 0 and expose:
                print('Done with computing the uncertainties in {}sec - {}/{}'.format(time.time() - prev_time, i,
                                                                                      nt, flush=True))
        fx_test_ts = np.concatenate(fx_test_ts, axis=0)
        expected_uncertainties = np.array(expected_uncertainties)

        if dpp:
            prev_time = time.time()
            sim_matrix = compute_dpp_sim_matrix(fx_test_ts, induced_probs, nt, compute_dpp_batch_size())
            if expose:
                print('Computing dpp took {}sec'.format(time.time() - prev_time))
        else:
            sim_matrix = fx_test_ts

        return expected_uncertainties, original_fx_test_t, sim_matrix

    parallel_data_prep = prepare_data_for_p_compute

    gpu_p_compute_batch_uncertainty = pmap(v_compute_batch_uncertainty,in_axes=tuple([0] * 6 + [None] * 10),
                                           static_broadcasted_argnums=(7, 8, 12, 13), backend=backend)

    def delete_cpu_cache():
        try:
            predict_normal_mse._clear_cache()
            jv_compute_batch_uncertainty._clear_cache()
        except:
            pass

    if ntk_config.get('sequential_selection', False):
        print('Sequential selection detected.')
        selections = []
        add_num = al_params['add_num']

        p = add_num + n
        for i in range(add_num):
            prev_time = time.time()
            print('Min eigenvalue: ', np.linalg.eigvals(
                k_train_train.transpose(0, 2, 1, 3).reshape(n * num_classes, n * num_classes)).min())
            expected_uncertainties, original_fx_test_t, fx_test_ts = compute_uncertainties(
                nt, n, k_train_train, k_test_train, k_test_test, induced_probs, fx_train_0,
                fx_test_0, y_onehots, y_train_onehot, num_classes, learning_rate, num_devices,
                p_compute_batch_uncertainty, jv_compute_batch_uncertainty,
                gpu_p_compute_batch_uncertainty, parallel_data_prep, pad=True, p=p, expose=False,
                selections=selections)
            point = select_top_point_according_to_objective(fx_test_ts, original_fx_test_t, y_test, selections)
            k_point = k_test_test[point:point + 1, point:point + 1]
            k_row = k_test_train[point:point + 1]
            k_train_train = np.concatenate((np.concatenate((k_train_train, k_row), axis=0),
                                            np.concatenate((k_row, k_point), axis=1).transpose(1, 0, 3, 2)), axis=1)
            # nt x n x k x k --> nt x (n+1) x k x k
            k_row = k_test_test[:, point:point + 1]
            k_test_train = np.concatenate((k_test_train, k_row), axis=1)
            fx_train_0 = np.concatenate((fx_train_0, fx_test_0[point:point + 1]), axis=0)
            new_y_train_point = np.zeros((1, num_classes))
            if ntk_config.get('sequential_with_true_label', True):
                new_y_train_point[0, y_test[point]] = 1.
            else:
                new_y_train_point[0, np.argmax(np.asarray(original_fx_test_t)[point])] = 1.
            y_train_onehot = np.concatenate((y_train_onehot, new_y_train_point), axis=0)
            n += 1
            selections.append(point)
            print('Selected point - {} from class - {} with uncrt - {}, time: {}'.format(
                point, y_test[point], expected_uncertainties[point], time.time() - prev_time))
        sorted_points = np.array([i for i in np.arange(nt) if i not in selections] + selections[::-1])

    elif ntk_config.get('hybrid_selection', False):
        print('Hybrid selection detected.')
        c = ntk_config['hybrid_c']
        add_num = al_params['add_num']
        hybrid_subsetting_method = ntk_config.get('hybrid_subsetting', 'batch')

        selections = []
        hybrid_prev_time = time.time()
        for i in range(int(add_num // c)):
            p = n + (i + 1) * c
            sub_selections = []
            if hybrid_subsetting_method == 'sequential':
                for j in range(c):
                    prev_time = time.time()
                    expected_uncertainties, original_fx_test_t, dpp_sim_matrix = compute_uncertainties(
                        nt, n, k_train_train, k_test_train, k_test_test, induced_probs, fx_train_0,
                        fx_test_0, y_onehots, y_train_onehot, num_classes, learning_rate, num_devices,
                        p_compute_batch_uncertainty, jv_compute_batch_uncertainty,
                        gpu_p_compute_batch_uncertainty, parallel_data_prep, pad=True, p=p, expose=False)

                    sorted_e_uncertainties = expected_uncertainties.argsort()[::-1]
                    point = next(i for i in sorted_e_uncertainties
                                 if i not in selections and i not in sub_selections and not np.isnan(
                        expected_uncertainties[i]))
                    # n x n x k x k --> (n+1) x (n+1) x k x k
                    k_point = k_test_test[point:point + 1, point:point + 1]
                    k_row = k_test_train[point:point + 1]
                    k_train_train = np.concatenate(
                        (np.concatenate((k_train_train, k_row), axis=0),
                         np.concatenate((k_row, k_point), axis=1).transpose(1, 0, 3, 2)), axis=1)
                    # nt x n x k x k --> nt x (n+1) x k x k
                    k_row = k_test_test[:, point:point + 1]
                    k_test_train = np.concatenate((k_test_train, k_row), axis=1)
                    fx_train_0 = np.concatenate((fx_train_0, fx_test_0[point:point + 1]), axis=0)
                    new_y_train_point = np.zeros((1, num_classes))
                    new_y_train_point[0, np.argmax(np.asarray(original_fx_test_t)[point])] = 1.
                    y_train_onehot = np.concatenate((y_train_onehot, new_y_train_point), axis=0)
                    n += 1
                    sub_selections.append(point)
                    print('Selected point - {} from class - {} with uncrt - {}, time: {}'.format(
                        point, y_test[point], expected_uncertainties[point], time.time() - prev_time))
                sub_selections = np.array(sub_selections)
            elif hybrid_subsetting_method == 'batch':
                if c <= 1:
                    raise ValueError('Hybrid_C should be > 1 when using batch subset selection method!')
                prev_time = time.time()
                expected_uncertainties, original_fx_test_t, dpp_sim_matrix = compute_uncertainties(
                    nt, n, k_train_train, k_test_train, k_test_test, induced_probs, fx_train_0, fx_test_0, y_onehots,
                    y_train_onehot,
                    num_classes, learning_rate, num_devices, p_compute_batch_uncertainty, jv_compute_batch_uncertainty,
                    gpu_p_compute_batch_uncertainty, parallel_data_prep, expose=True, dpp=use_dpp)
                if use_dpp:
                    mask = np.ones(nt, bool)
                    mask[selections] = False
                    masked_sim_matrix = dpp_sim_matrix[np.ix_(mask, mask)]
                    masked_e_uncertainties = expected_uncertainties[mask]
                    masked_sorted_points = get_sorted_points_with_dpp(
                        masked_e_uncertainties, masked_sim_matrix, nt, ntk_config['gamma'], c)
                    # This doesn't preserve order! but works as order doesn't really matter here.
                    sorted_points = np.in1d(expected_uncertainties,
                                            masked_e_uncertainties[masked_sorted_points]).nonzero()[0]
                else:
                    add_num = al_params['add_num']
                    sorted_points = select_batch_according_to_objective(
                        dpp_sim_matrix, original_fx_test_t, y_test, add_num)[::-1].copy()
                sub_selections = np.array(list(filter(lambda x: x not in selections, sorted_points[::-1])))[:c]
                k_point = k_test_test[np.ix_(sub_selections, sub_selections)]
                k_row = k_test_train[sub_selections]
                # n x n x k x k --> (n+c) x (n+c) x k x k
                k_train_train = np.concatenate(
                    (np.concatenate((k_train_train, k_row), axis=0),
                     np.concatenate((k_row, k_point), axis=1).transpose(1, 0, 3, 2)), axis=1)
                # nt x n x k x k --> nt x (n+c) x k x k
                k_row = k_test_test[:, sub_selections]
                k_test_train = np.concatenate((k_test_train, k_row), axis=1)
                fx_train_0 = np.concatenate((fx_train_0, fx_test_0[sub_selections]), axis=0)
                new_y_train_point = np.zeros((c, num_classes))
                selected_classes = []
                for z, point in enumerate(sub_selections):
                    selected_classes.append(np.argmax(np.asarray(original_fx_test_t)[point]))
                    new_y_train_point[z, selected_classes[-1]] = 1.
                y_train_onehot = np.concatenate((y_train_onehot, new_y_train_point), axis=0)
                n += c
                print('Selected {} points with class distribution {} - time: {}sec'.format(
                    c, str(dict(Counter(selected_classes))), time.time() - prev_time))
            else:
                raise ValueError(
                    'Subset selection method {} is not supported in hybrid mode!'.format(hybrid_subsetting_method))
            original_y_train = y_train_onehot[:-c]
            selected_y_train = y_test_onehot[sub_selections]
            y_train_onehot = np.concatenate((original_y_train, selected_y_train), axis=0)
            selections.extend(sub_selections)
            xla._xla_callable.cache_clear()
            delete_cpu_cache()
        sorted_points = np.array([i for i in np.arange(nt) if i not in selections] + selections[::-1])
        print('Hybrid selection finished with time: {}sec.'.format(time.time() - hybrid_prev_time))
    else:

        print('Batch selection {} DPP detected.'.format('with' if use_dpp else 'without'))
        pseudo_label_strategy = ntk_config.get('pseudo_label_strategy', 'torch')
        pseudo_labels = None
        if pseudo_label_strategy == 'torch':
            pseudo_labels = fx_test_0.argmax(axis=1)
        prev_time = time.time()
        expected_uncertainties, ntk_fx_test_inf, dpp_sim_matrix = compute_uncertainties(
            nt, n, k_train_train, k_test_train, k_test_test, induced_probs, fx_train_0, fx_test_0, y_onehots,
            y_train_onehot,
            num_classes, learning_rate, num_devices, p_compute_batch_uncertainty, jv_compute_batch_uncertainty,
            gpu_p_compute_batch_uncertainty, parallel_data_prep, expose=True, dpp=use_dpp, pseudo_labels=pseudo_labels)
        if use_dpp:
            sorted_points = get_sorted_points_with_dpp(
                expected_uncertainties, dpp_sim_matrix, nt, ntk_config['gamma'], al_params['add_num'])
        else:
            add_num = al_params['add_num']
            initial_fx = fx_test_0 if pseudo_label_strategy == 'torch' else ntk_fx_test_inf
            sorted_points = select_batch_according_to_objective(dpp_sim_matrix, initial_fx, y_test, add_num)[::-1].copy()

        if ntk_config.get('use_kmeans', False):
            print('Using KMeans++')
            extended_fx_test_0 = fx_test_0.reshape((1, nt, -1))
            distributions = np.reshape(dpp_sim_matrix - extended_fx_test_0, (nt, -1))
            points = init_centers(distributions, add_num)
            sorted_points = np.array(list(set(np.arange(len(subset))) - set(points)) + points)

        if eps > 0.:
            r_count = 0
            random_selections = np.random.choice(nt-add_num, add_num, False)
            for i in range(add_num):
                if np.random.random() < eps:
                    tmp = sorted_points[-i-1]  # point to be randomized
                    index_to_be_replaced = np.argwhere(sorted_points == random_selections[i])[0, 0]
                    sorted_points[-i-1] = random_selections[i]  # random point to fill in instead of point
                    sorted_points[index_to_be_replaced] = tmp  # the poor datapoint that was selected at the beginning
                    r_count += 1
            print(f'{r_count} points were randomly selected!')

        print('Batch selection finished in time: {}, number of nans: {}'.format(time.time() - prev_time,
                                                                                np.isnan(sorted_points).sum()))

    xla._xla_callable.cache_clear()
    delete_cpu_cache()
    print('Live buffers count:', len(backend.live_buffers()))
    print('Live executables count:', len(backend.live_executables()))

    return sorted_points