from src.core.checkpointers import CustomCheckpointer

def build(save_dir, logger, model, optimizer, scheduler, eval_standard, init=False):
    checkpointer = CustomCheckpointer(
        save_dir, logger, model, optimizer, scheduler, eval_standard, init=init)
    return checkpointer

