from src.core.evaluators import AccEvaluator

EVAL_TYPES = {
    'acc': AccEvaluator,
}

def build(eval_config, logger):
    standard = eval_config['standard']
    evaluator = EVAL_TYPES[standard](logger)

    logger.infov('Evaluator is build.')
    return evaluator
