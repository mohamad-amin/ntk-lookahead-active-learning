from src.core.criterions import CustomCriterion


def build(train_config, model_config, logger):
    criterion_params = train_config.get('criterion', {})
    criterion_params['num_classes'] = model_config['model_arch']['num_classes']
    criterion = CustomCriterion(**criterion_params)

    if logger is not None:
        logger.infov('Criterion is built.')
    return criterion
