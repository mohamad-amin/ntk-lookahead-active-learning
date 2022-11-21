from torch import optim
from copy import deepcopy

OPTIMIZERS = {
    'sgd': optim.SGD,
    'rmsprop': optim.RMSprop,
    'adam': optim.Adam,
}

def build(optim_config, model_params, logger):
    optim_params = deepcopy(optim_config)
    optimizer_name = optim_params.pop('name')
    optim_params['params'] = model_params

    if optimizer_name in OPTIMIZERS:
        optimizer = OPTIMIZERS[optimizer_name](**optim_params)
    else:
        logger.error(
            'Specify a valid optimizer name among {}.'.format(OPTIMIZERS.keys())
        ); exit()

    if logger is not None:
        logger.infov('{} opimizer is built.'.format(optimizer_name.upper()))
    return optimizer
