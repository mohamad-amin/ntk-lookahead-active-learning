from src.core.meters import AverageEpochMeter, PrecisionRecallMeter

def build(model_config, logger):
    loss_meter = AverageEpochMeter('loss meter', logger, fmt=':f')
    num_classes = model_config['model_arch']['num_classes']
    pr_meter = PrecisionRecallMeter('pr meter', logger, num_classes, fmt=':f')

    logger.infov('Loss and PR meters are built.')
    return loss_meter, pr_meter
