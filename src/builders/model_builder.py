import jax
import jax.numpy as jnp
import numpy as np
import neural_tangents as nt
from functools import partial
from src.core.models import wide_resnet_jax, wide_resnet, resnet_jax, resnet, lossnet, testnet
from src.utils.ntk_util import get_ntk_input_shape
from src.utils.empirical import empirical_ntk_fn
from jax import jit
from src.utils import util

MODELS = {
    'wide_resnet': wide_resnet.Wide_ResNet,
    'resnet18': resnet.ResNet18,
    'resnet34': resnet.ResNet34,
    'testnet': testnet.TestNet,
}

NTK_MODELS = {
    'wide_resnet_wo': wide_resnet.Wide_Resnet_NTK,
    'wide_resnet_w': wide_resnet_jax.WideResNetNTK,
    'resnet18_w': resnet_jax.ResNet18,
    'resnet34_w': resnet_jax.ResNet34,
    'testnet_w': testnet.TestNetNTK,
}

LOSS_MODELS = {
    'lossnet': lossnet.LossNet
}


def build(model_config, data_config, logger):
    backbone = model_config['backbone']
    model_arch = model_config['model_arch']
    acquisition = model_config['acquisition']
    ntk_config = model_config.get('ntk', {})
    data_name = data_config['name']
    num_classes = model_arch['num_classes']
    bn_with_running_stats = model_arch.pop('bn_with_running_stats', True)

    if data_name in ['cifar10', 'svhn', 'cifar100']:
        num_input_channels = 3
    else:
        num_input_channels = 1
    model_arch['num_input_channels'] = num_input_channels

    # Build a model
    models = {}
    if backbone in MODELS:
        model = MODELS[backbone](**model_arch)
        models['model'] = model
        models['use_ntk'] = False
        logger.info(
            'A model {} is built.'.format(backbone))

        if 'eer' in acquisition:
            retrain_model = MODELS[backbone](**model_arch)
            models['retrain_model'] = retrain_model

        if 'ntk' in acquisition:
            with_or_without = '_w' if bn_with_running_stats else '_wo'
            backbone = backbone + with_or_without

            if bn_with_running_stats:
                ntk_model = NTK_MODELS[backbone](**model_arch)
                init_fn, apply_fn = ntk_model.init, ntk_model.apply
                apply_fn = partial(apply_fn, mutable=False)
            else:
                init_fn, apply_fn, _ = NTK_MODELS[backbone](**model_arch)
                ntk_model = None

            # Define a loss function
            loss_fn_name = ntk_config.get('loss_fn', 'cross_entropy')
            loss_fn = None
            if loss_fn_name == 'cross_entropy':
                loss_fn = lambda fx, y_hat: -jnp.mean(jax.experimental.stax.logsoftmax(fx) * y_hat)

            ntk_implementation = ntk_config.get('kernel_implementation', 3)
            print(f'Kernel implementation: {ntk_implementation}')

            # Initialize ntk params
            rng = jax.random.PRNGKey(ntk_config['seed'])

            if not bn_with_running_stats:
                apply_fn = partial(apply_fn, **{'rng': rng})

            trace_axes = ntk_config.get('trace_axes', [])

            def apply_fn_trace(params, x):
                out = apply_fn(params, x)
                return np.sum(out, axis=-1) / out.shape[-1] ** 0.5
                # return out[:, 0]

            if len(trace_axes) > 0 and trace_axes[0] == -1:
                apply_fn_ntk = apply_fn_trace
            else:
                apply_fn_ntk = apply_fn

            ntk_fn = empirical_ntk_fn(
                apply_fn_ntk, vmap_axes=0, implementation=ntk_implementation, trace_axes=())
            ntk_trace_fn = empirical_ntk_fn(
                apply_fn, vmap_axes=0, implementation=ntk_implementation, trace_axes=(-1,))
            ntk_fn_batched = nt.batch(
                ntk_fn, device_count=-1, batch_size=ntk_config['ntk_batch'], store_on_device=False)
            ntk_fn_batch_builder = partial(nt.batch, kernel_fn=ntk_fn, device_count=-1, store_on_device=False)
            ntk_trace_fn_batch_builder = partial(nt.batch, kernel_fn=ntk_trace_fn, device_count=-1, store_on_device=False)

            if bn_with_running_stats:
                ntk_params = init_fn(rng, jnp.ones(get_ntk_input_shape(data_config, num_input_channels)))
            else:
                _, ntk_params = init_fn(rng, get_ntk_input_shape(data_config, num_input_channels, old=True))

            single_device_ntk_fn = partial(nt.batch, kernel_fn=ntk_fn, device_count=1, store_on_device=False)

            models.update({
                'ntk_model': ntk_model,
                'ntk_fn': ntk_fn_batched,
                'ntk_fn_builder': ntk_fn_batch_builder,
                'ntk_trace_fn_builder': ntk_trace_fn_batch_builder,
                'ntk_params': ntk_params,
                'ntk_params_size': util.get_params_size(ntk_params),
                'single_device_ntk_fn': single_device_ntk_fn,
                'apply_fn': apply_fn,
                'loss_fn': loss_fn,
                'rng': rng,
                'use_ntk': True
            })
            logger.info(
                'A NTK model {} is built.'.format(backbone))

        if 'll4al' in acquisition:
            num_layers = model_arch['num_layers']
            if 'wide' in backbone:
                widen_factor = model_arch['widen_factor']
                feature_sizes = [28, 14, 7][:num_layers]
                num_channels = [16*widen_factor, 32*widen_factor, 64*widen_factor][:num_layers]
            else:
                feature_sizes=[32, 16, 8, 4]
                num_channels=[64, 128, 256, 512]

            models['loss_model'] = LOSS_MODELS['lossnet'](
                num_layers=num_layers, feature_sizes=feature_sizes, num_channels=num_channels)

    else:
        logger.error(
            'Specify valid backbone or model type among {}.'.format(MODELS.keys())
        ); exit()

    return models

