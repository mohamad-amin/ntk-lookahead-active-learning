from copy import deepcopy
from torch import optim
from src.core.schedulers import CustomScheduler


SCHEDULERS = {
    'multi': optim.lr_scheduler.MultiStepLR,
    'onecycle': optim.lr_scheduler.OneCycleLR,
    'custom': CustomScheduler
}

def build(train_config, optimizer, logger, epochs, steps_per_epoch):
    if 'lr_schedule' not in train_config:
        logger.warn('No scheduler is specified.')
        return None

    schedule_config = deepcopy(train_config['lr_schedule'])
    scheduler_name = schedule_config.pop('name', 'multi_step')
    schedule_config['optimizer'] = optimizer

    if scheduler_name == 'onecycle':
        schedule_config['epochs'] = epochs
        schedule_config['steps_per_epoch'] = steps_per_epoch
    else:
        if scheduler_name == 'custom' and schedule_config.get('milestones', None) is None:
            schedule_config['milestones'] = [int(train_config['num_epochs'] * 0.8)]
            if schedule_config.get('max_lr', None) is not None:
                schedule_config.pop('max_lr')

    if scheduler_name in SCHEDULERS:
        scheduler = SCHEDULERS[scheduler_name](**schedule_config)
    else:
        logger.error(
            'Specify a valid scheduler name among {}.'.format(SCHEDULERS.keys())
        ); exit()

    logger.infov('{} scheduler is built.'.format(scheduler_name.upper()))
    return scheduler
