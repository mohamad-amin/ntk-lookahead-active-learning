import os
import time
import torch
from collections import OrderedDict


class CustomCheckpointer(object):
    def __init__(self, checkpoint_dir, logger, model,
                 optimizer, scheduler, eval_standard='accuracy', init=False):
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if init:
            best_checkpoint_path = os.path.join(
                self.checkpoint_dir, 'best_checkpoint')
            if os.path.exists(best_checkpoint_path):
                os.remove(best_checkpoint_path)
            last_checkpoint_path = os.path.join(
                self.checkpoint_dir, 'last_checkpoint')
            if os.path.exists(last_checkpoint_path):
                os.remove(last_checkpoint_path)
            al_results_path = os.path.join(
                self.checkpoint_dir, 'al_results')
            if os.path.exists(al_results_path):
                os.remove(al_results_path)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.eval_standard = eval_standard
        self.reset()

        self.logger.infov('Checkpointer is built.')

    def reset(self):
        self.best_eval_metric = 0.
        self.last_eval_metric = 0.

    def update(self, eval_metrics):
        eval_metric = eval_metrics[self.eval_standard]
        self.last_eval_metric = eval_metric
        if eval_metric > self.best_eval_metric:
            self.best_eval_metric = eval_metric

    def save(self, epoch, num_steps, cycle=0, trial=0, eval_metrics=None):
        # Save the given checkpoint in train time and best checkpoint in eval time.
        if eval_metrics is None:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 'checkpoint' + '_' + str(trial) + '_' +  str(epoch) + '_' + str(num_steps) + '.pth')
        else:
            eval_metric = eval_metrics[self.eval_standard]
            self.last_eval_metric = eval_metric
            if eval_metric <= self.best_eval_metric:
                return
            self.best_eval_metric = eval_metric
            checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth'.format(trial))

        model_params = {'trial': trial, 'epoch': epoch, 'num_steps': num_steps}
        if isinstance(self.model, torch.nn.DataParallel):
            model_params['state_dict'] = self.model.module.state_dict()
        else:
            model_params['state_dict'] = self.model.state_dict()

        model_params['optimizer_state_dict'] = self.optimizer.state_dict()
        model_params['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(model_params, checkpoint_path)
        if 'best' in checkpoint_path:
            torch.save(model_params, os.path.join(self.checkpoint_dir, 'checkpoint_best_{}.pth'.format(cycle)))

        self.logger.info(
            'A checkpoint is saved for trial={}, cycle={}, epoch={}, steps={}.'.format(
                trial, cycle, epoch, num_steps))

        # Update the checkpoint record
        if eval_metrics is not None:
            best_checkpoint_info = {'trial': trial, 'cyclec': cycle, 'epoch': epoch, 'num_steps': num_steps}
            best_checkpoint_info.update(eval_metrics)

            self._record_best_checkpoint(best_checkpoint_info, trial=trial)
        else:
            self._record_last_checkpoint(checkpoint_path, trial=trial)

    def load(self, mode, checkpoint_path=None, trial=0, use_latest=True):
        strict = True
        if mode == 'train':
            if self._has_checkpoint() and use_latest: # Override argument with existing checkpoint
                checkpoint_path = self._get_checkpoint_path(trial)
            if not checkpoint_path:
                self.logger.info("No checkpoint found. Initializing model from scratch.")
                return {}
        else:
            if not checkpoint_path:
                while not self._has_checkpoint():
                    self.logger.warn('No checkpoint available. Wait for 60 seconds.')
                    time.sleep(60)
                checkpoint_path = self._get_checkpoint_path(trial)

        self.logger.info("Loading checkpoint from {}".format(checkpoint_path))
        checkpoint_dict = self._load_checkpoint(checkpoint_path)

        self.model.load_state_dict(
            checkpoint_dict.pop('state_dict'), strict=strict)

        if strict:
            if 'optimizer_state_dict' in checkpoint_dict and self.optimizer:
                self.logger.info("Loading optimizer from {}".format(checkpoint_path))
                self.optimizer.load_state_dict(checkpoint_dict.pop('optimizer_state_dict'))
            if 'scheduler_state_dict' in checkpoint_dict and self.scheduler:
                self.logger.info("Loading scheduler from {}".format(checkpoint_path))
                self.scheduler.load_state_dict(checkpoint_dict.pop('scheduler_state_dict'))

        return checkpoint_dict

    def record_results(self, trial, cycle):
        record_path = os.path.join(self.checkpoint_dir, "al_results")
        with open(record_path, "a+") as f:
            f.seek(0)
            data = f.read(trial * 10 + cycle + 100) # To move to the last line of the file
            if len(data) > 0:
                f.write("\n")
            f.write("[Trial {}, Cycle {}] {}: {:4f} - {:4f}".format(
                trial, cycle, self.eval_standard, self.best_eval_metric, self.last_eval_metric))


    def _freeze(self):
        for param in self.model.layers.parameters():
            param.requires_grad = False

    def _has_checkpoint(self):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        return os.path.exists(record_path)

    def _get_checkpoint_path(self, trial):
        record_path = os.path.join(self.checkpoint_dir, "last_checkpoint")
        try:
            with open(record_path, "r") as f:
                last_saved = f.readlines()[trial]
                last_saved = last_saved.strip()
        except IOError:
            self.logger.warn('If last_checkpoint file doesn not exist, maybe because \
                              it has just been deleted by a separate process.')
            last_saved = ''
        return last_saved

    def _record_last_checkpoint(self, last_checkpoint_path, trial):
        record_path = os.path.join(self.checkpoint_dir, 'last_checkpoint')
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                list_of_lines = f.readlines()

            if list_of_lines:
                try:
                    list_of_lines[trial] = last_checkpoint_path
                except:
                    list_of_lines += [last_checkpoint_path]

                with open(record_path, 'w') as f:
                    f.writelines(str(list_of_lines))
                    return

        with open(record_path, 'w') as f:
            f.write(str(last_checkpoint_path))

    def _record_best_checkpoint(self, best_checkpoint_info, trial):
        record_path = os.path.join(self.checkpoint_dir, 'best_checkpoint')
        if os.path.exists(record_path):
            with open(record_path, 'r') as f:
                list_of_lines = f.readlines()

            if list_of_lines:
                try:
                    list_of_lines[trial] = best_checkpoint_info
                except:
                    list_of_lines += [best_checkpoint_info]

                with open(record_path, 'w') as f:
                    f.writelines(str(list_of_lines))
                    return

        with open(record_path, 'w') as f:
            f.write(str(best_checkpoint_info))


    def _load_checkpoint(self, checkpoint_path):
        checkpoint_dict = torch.load(checkpoint_path)
        if torch.cuda.device_count() > 1:
            checkpoint = checkpoint_dict['state_dict']
            checkpoint = OrderedDict([('module.'+ k, v) for k, v in checkpoint.items()])
            checkpoint_dict['state_dict'] = checkpoint

        checkpoint_dict['checkpoint_path'] = checkpoint_path

        return checkpoint_dict
