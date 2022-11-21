import os
import jax
import torch
import time
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

from src.utils import util, ntk_util, al_util, batchnorm_utils
from src.builders import model_builder, dataloader_builder, checkpointer_builder,\
                         optimizer_builder, criterion_builder, scheduler_builder,\
                         meter_builder, evaluator_builder
from src.utils.probability_utils import project_into_probability_simplex, calc_MI_for_pairwise_features

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = '.80'


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger
        self.logger = logger

        # Load configurations
        config = util.load_config(config_path)

        self.model_config = config['model']
        self.train_config = config['train']
        self.eval_config = config['eval']
        self.data_config = config['data']

        self.eval_standard = self.eval_config['standard']

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()

        if device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('GPU is available with {} devices.'.format(self.num_devices))
        self.logger.warn('CPU is available with {} devices.'.format(jax.device_count('cpu')))

        # Load a summary writer
        self.save_dir = save_dir
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self):
        pass

    def evaluate(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _use_data_parallel(self):
        return torch.cuda.device_count() > 1 and self.torch_multi_gpu

    def _build(self, mode, init=False):

        # Build a dataloader
        self.dataloaders = dataloader_builder.build(
            self.data_config, self.logger)

        # Build a model
        self.models = model_builder.build(
            deepcopy(self.model_config), self.data_config, self.logger)

        # Determine acquisition strategy
        if 'acquisition' in self.model_config:
            self.acquisition = self.model_config['acquisition']
        else:
            self.acquisition = 'ntk' if self.models['use_ntk'] else 'random'

        self.torch_multi_gpu = self.model_config.get('torch_multi_gpu', True)

        # Use multi GPUs if available
        if not isinstance(self.models['model'], torch.nn.DataParallel):
            if self._use_data_parallel():
                self.models['model'] = util.DataParallel(self.models['model'])
            self.models['model'] = self.models['model'].to(self.device)
            if self.acquisition == 'll4al':
                if self._use_data_parallel():
                    self.models['loss_model'] = util.DataParallel(self.models['loss_model'])
                self.models['loss_model'] = self.models['loss_model'].to(self.device)

        # Build an optimizer, scheduler and criterion
        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.models['model'].parameters(), self.logger)
        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger,
            self.train_config['num_epochs'], len(self.dataloaders['train']))
        self.criterion = criterion_builder.build(
            self.train_config, self.model_config, self.logger)
        self.loss_meter, self.pr_meter = meter_builder.build(
            self.model_config, self.logger)
        self.evaluator = evaluator_builder.build(
            self.eval_config, self.logger)
        if self.acquisition == 'll4al':
            self.loss_optimizer = optimizer_builder.build(
                self.train_config['optimizer'], self.models['loss_model'].parameters(), self.logger)
            self.loss_scheduler = scheduler_builder.build(
                self.train_config, self.loss_optimizer, self.logger,
                self.train_config['num_epochs'], len(self.dataloaders['train']))

        # Build a checkpointer
        self.checkpointer = checkpointer_builder.build(
            self.save_dir, self.logger, self.models['model'], self.optimizer,
            self.scheduler, self.eval_standard, init=init)
        checkpoint_path = self.model_config.get('checkpoint_path', '')
        self.misc = self.checkpointer.load(
            mode=mode, checkpoint_path=checkpoint_path, use_latest=False)

        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_initial.pth')
        model_params = {'trial': 0}
        if self._use_data_parallel():
            model_params['state_dict'] = self.models['model'].module.state_dict()
        else:
            model_params['state_dict'] = self.models['model'].state_dict()
        torch.save(model_params, checkpoint_path)

    def run(self):
        trials = self.train_config.get('trials', 1)
        cycles = self.train_config.get('cycles', 10)
        query_first = self.train_config.get('query_first', False)

        for trial in range(trials):
            # Build components
            init = True if trial == 0 else False
            self._build(mode='train', init=init)

            for cycle in range(cycles):
                if query_first:
                    self._update_data(trial, cycle)

                # Train a model
                self._train(trial, cycle)
                self.checkpointer.record_results(trial, cycle)

                # Query new data points and update labeled and unlabeled pools
                if not query_first:
                    self._update_data(trial, cycle)

                self.checkpointer.reset()

    def _train(self, trial, cycle):
        start_epoch, num_steps = 0, 0
        num_epochs = self.train_config.get('num_epochs', 200)

        self.logger.info(
            'Trial {}, Cycle - {} - train for {} epochs starting from epoch {}'.format(
                trial, cycle, num_epochs, start_epoch))

        if self.train_config.get('manual_train_control', False):
            print('Training...')
            import IPython; IPython.embed()

        # Start training
        for epoch in range(start_epoch, start_epoch + num_epochs):
            train_start = time.time()
            num_steps = self._train_one_epoch(epoch, num_steps)
            train_time = time.time() - train_start

            lr = self.scheduler.get_lr()[0]

            self.logger.infov(
                '[Cycle {}, Epoch {}] with lr: {:5f} completed in {:3f} - train loss: {:4f}'\
                .format(cycle, epoch, lr, train_time, self.loss_meter.avg))
            self.writer.add_scalar('Train/learning_rate', lr, global_step=num_steps)

            if not self.train_config['lr_schedule']['name'] in ['onecycle']:
                self.scheduler.step()
                if self.acquisition == 'll4al':
                    self.loss_scheduler.step()

            self.loss_meter.reset()

            if epoch - start_epoch > 0.8 * num_epochs:
                is_last_epoch = start_epoch + num_epochs == epoch + 1
                eval_metrics = self._evaluate_once(trial, cycle, epoch, num_steps, is_last_epoch)
                self.checkpointer.save(epoch, num_steps, cycle, trial, eval_metrics)
                self.logger.info(
                    '[Epoch {}] - {}: {:4f}'.format(
                        epoch, self.eval_standard, eval_metrics[self.eval_standard]))
                self.logger.info(
                    '[Epoch {}] - best {}: {:4f}'.format(
                        epoch, self.eval_standard, self.checkpointer.best_eval_metric))

        if self.train_config.get('adjust_batchnorm_stats', True):
            self._adjust_batchnorm_to_population()
            eval_metrics = self._evaluate_once(trial, cycle, start_epoch + num_epochs, num_steps, True)
            self.checkpointer.save(start_epoch + num_epochs, num_steps, cycle, trial, eval_metrics)
            self.logger.info(
                'After BN adjust - {}: {:4f}'.format(self.eval_standard, eval_metrics[self.eval_standard]))
            self.logger.info(
                'After BN adjust - best {}: {:4f}'.format(self.eval_standard, self.checkpointer.best_eval_metric))

    def _train_one_epoch(self, epoch, num_steps):

        self.models['model'].train()
        if self.acquisition == 'll4al': self.models['loss_model'].train()
        dataloader = self.dataloaders['train']
        num_batches = len(dataloader)

        for i, input_dict in enumerate(dataloader):
            input_dict = util.to_device(input_dict, self.device)

            # Forward propagation
            self.optimizer.zero_grad()
            if self.acquisition == 'll4al':
                self.loss_optimizer.zero_grad()
            output_dict = self.models['model'](input_dict)

            # Compute losses
            output_dict['labels'] = input_dict['labels']

            if self.acquisition == 'll4al':
                features = output_dict['features']
                loss_num_epochs = self.train_config.get(
                    'loss_num_epochs', self.train_config['num_epochs'] * 0.6)
                if epoch > loss_num_epochs:
                    for feature_idx in range(len(features)):
                        features[feature_idx] = features[feature_idx].detach()
                pred_loss = self.models['loss_model'](features)
                pred_loss = pred_loss.view(pred_loss.size(0))
                output_dict['pred_loss'] = pred_loss

            losses = self.criterion(output_dict)
            loss = losses['loss']

            # Backward propagation
            loss.backward()
            self.optimizer.step()
            if self.acquisition == 'll4al':
                self.loss_optimizer.step()

            # Print losses
            batch_size = input_dict['inputs'].size(0)
            self.loss_meter.update(loss.item(), batch_size)
            if i % (len(dataloader) / 10) == 0:
                self.loss_meter.print_log(epoch, i, num_batches)

            # step scheduler if needed
            if self.train_config['lr_schedule']['name'] in ['onecycle']:
                self.scheduler.step()
                if self.acquisition == 'll4al':
                    self.loss_scheduler.step()

            # Save a checkpoint
            num_steps += batch_size

        return num_steps

    def evaluate(self):
        def _get_misc_info(misc):
            infos = ['epoch', 'num_steps', 'checkpoint_path']
            return (misc[info] for info in infos)

        self._build(mode='eval')
        epoch, num_steps, current_checkpoint_path = _get_misc_info(self.misc)
        last_evaluated_checkpoint_path = None
        while True:
            if last_evaluated_checkpoint_path == current_checkpoint_path:
                self.logger.warn('Found already evaluated checkpoint. Will try again in 60 seconds.')
                time.sleep(60)
            else:
                eval_metrics = self._evaluate_once(epoch, num_steps)
                last_evaluated_checkpoint_path = current_checkpoint_path
                self.checkpointer.save(
                    epoch, num_steps, eval_metrics=eval_metrics)

            # Reload a checkpoint. Break if file path was given as checkpoint path.
            checkpoint_path = self.model_config.get('checkpoint_path', '')
            if os.path.isfile(checkpoint_path): break
            misc = self.checkpointer.load(checkpoint_path, use_latest=True)
            epoch, num_step, current_checkpoint_path = _get_misc_info(misc)

    def _evaluate_once(self, trial, cycle, epoch, num_steps, is_last_epoch=False):
        dataloader = self.dataloaders['val']

        self.models['model'].eval()
        self.logger.info('[Cycle {} Epoch {}] Evaluating...'.format(cycle, epoch))
        labels = []
        outputs = []

        for input_dict in dataloader:
            with torch.no_grad():
                input_dict = util.to_device(input_dict, self.device)
                # Forward propagation
                output_dict = self.models['model'](input_dict)
                output_dict['labels'] = input_dict['labels']
                labels.append(input_dict['labels'])
                outputs.append(output_dict['logits'])

        output_dict = {
            'logits': torch.cat(outputs),
            'labels': torch.cat(labels)
        }

        if is_last_epoch and False:
            probs = project_into_probability_simplex(output_dict['logits'].detach().cpu().numpy())
            mis = calc_MI_for_pairwise_features(probs)
            print('Mutual information table:')
            print(pd.DataFrame(mis))
            print('Norm of them:', np.linalg.norm(mis))

        # Print losses
        self.evaluator.update(output_dict)

        self.evaluator.print_log(epoch, num_steps)
        eval_metric = self.evaluator.compute()

        # Reset the evaluator
        self.evaluator.reset()
        return {self.eval_standard: eval_metric}

    def _forward_pass(self, partition='subset'):
        unlabeled_dataset = self.dataloaders['unlabeled'].dataset
        data = self.dataloaders[partition]
        X, _ = ntk_util.get_full_data(unlabeled_dataset, data)
        batch_size = X.shape[0]
        predictions = []
        with torch.no_grad():
            for i in range(0, batch_size, 1000):
                pred = self.models['model'](
                    {'inputs': X[i:i+1000].to(self.device)})['logits'].detach().cpu().numpy()
                predictions.append(pred)
        return np.concatenate(predictions)

    def _adjust_batchnorm_to_population(self):

        self.logger.info('Adjusting BatchNorm statistics to population values...')

        net = self.models['model']
        train_dataset = self.dataloaders['train'].dataset
        trainloader = torch.utils.data.DataLoader(train_dataset,
                                                 batch_size=self.data_config['batch_size'],
                                                 num_workers=self.data_config['num_workers'],
                                                 drop_last=True)

        net.apply(batchnorm_utils.adjust_bn_layers_to_compute_populatin_stats)
        for _ in range(3):
            with torch.no_grad():
                for input_dict in trainloader:
                    input_dict = util.to_device(input_dict, self.device)
                    net(input_dict)
        net.apply(batchnorm_utils.restore_original_settings_of_bn_layers)

        self.logger.info('BatchNorm statistics adjusted.')

    def _update_data(self, trial, cycle):
        al_params = self.data_config['al_params']
        subset = self.dataloaders['subset']

        checkpoint_path = os.path.join(self.save_dir, 'checkpoint_tmp.pth')
        model_params = {'trial': trial}
        if self._use_data_parallel():
            model_params['state_dict'] = self.models['model'].module.state_dict()
        else:
            model_params['state_dict'] = self.models['model'].state_dict()
        torch.save(model_params, checkpoint_path)
        del model_params

        acquisition_method = self.acquisition
        self.logger.info('Acquisition method: {}'.format(acquisition_method))

        if acquisition_method == 'ntk':

            bn_with_running_stats = self.model_config['model_arch'].get('bn_with_running_stats', True)
            self.models['ntk_params'] = ntk_util.update_ntk_params(
                self.models['ntk_params'], self.models['model'], bn_with_running_stats)

            subset_predictions = self._forward_pass()

            del self.models['model']
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()

            device_count = self.num_devices if torch.cuda.device_count() > 0 else 1
            cycle_count = self.train_config.get('cycles', 10)

            # Measure uncertainty of each data points in the subset
            block_computation = self.model_config['ntk'].get('block', False)
            permutation_batching = self.model_config['ntk'].get('permutation_batching', False)
            if not block_computation:
                raise NotImplementedError()
            else:
                if permutation_batching:
                    raise NotImplementedError()
                else:
                    uncertainty = ntk_util.get_uncertainty_ntk_block_sequential(
                        self.dataloaders, self.models, subset_predictions, self.model_config, al_params,
                        cycle, cycle_count, device_count)

            arg = uncertainty

        elif acquisition_method == 'eer':
            uncertainty = al_util.get_eer_parallel(self.dataloaders, self.models, self.model_config,
                                          self.train_config, self.data_config, self.logger, self.device, self.save_dir)
            arg = torch.argsort(torch.tensor(uncertainty)).cpu().numpy()
            torch.cuda.empty_cache()

        elif acquisition_method == 'entropy':

            dataloader = self.dataloaders['unlabeled']
            if not isinstance(self.models['model'], torch.nn.DataParallel):
                if self._use_data_parallel():
                    self.models['model'] = util.DataParallel(self.models['model'])
                self.models['model'].to(self.device)
            self.models['model'].eval()

            outputs = []
            for j, input_dict in enumerate(dataloader):
                with torch.no_grad():
                    input_dict = util.to_device(input_dict, self.device)
                    # Forward propagation
                    output_dict = self.models['model'](input_dict)
                    outputs.append(output_dict['logits'])

            # Compute entropy
            uncertainty = al_util.compute_entropy(torch.cat(outputs))

            arg = torch.argsort(torch.tensor(uncertainty)).cpu().numpy()
            torch.cuda.empty_cache()

        elif acquisition_method == 'll4al':
            dataloader = self.dataloaders['unlabeled']
            uncertainty = torch.tensor([])
            self.models['model'].eval()
            self.models['loss_model'].eval()
            for j, input_dict in enumerate(dataloader):
                with torch.no_grad():
                    input_dict = util.to_device(input_dict, self.device)

                    # Forward propagation
                    output_dict = self.models['model'](input_dict)
                    features = output_dict['features']
                    pred_loss = self.models['loss_model'](features)
                    pred_loss = pred_loss.view(pred_loss.size(0)).detach().cpu()
                    uncertainty = torch.cat((uncertainty, pred_loss), 0)

            arg = torch.argsort(uncertainty).cpu().numpy()
            torch.cuda.empty_cache()

        elif acquisition_method == 'margin':
            dataloader = self.dataloaders['unlabeled']
            uncertainty = []
            if not isinstance(self.models['model'], torch.nn.DataParallel):
                if self._use_data_parallel():
                    self.models['model'] = util.DataParallel(self.models['model'])
                self.models['model'].to(self.device)
            self.models['model'].eval()
            for j, input_dict in enumerate(dataloader):
                with torch.no_grad():
                    input_dict = util.to_device(input_dict, self.device)

                    # Forward propagation
                    output_dict = self.models['model'](input_dict)

                    # Compute entropy
                    margin = al_util.compute_margin(output_dict['logits'])
                    uncertainty.append(margin.item())

            arg = torch.argsort(torch.tensor(uncertainty)).cpu().numpy()
            torch.cuda.empty_cache()

        elif acquisition_method == 'badge':
            badge_sampler = al_util.BadgeSampling(
                self.models['model'], self.data_config, self.model_config)
            points = badge_sampler.query(
                n=al_params['add_num'], loader_te=self.dataloaders['unlabeled'])
            arg = np.array(list(set(np.arange(len(subset))) - set(points)) + points)
            torch.cuda.empty_cache()
        else:
            arg = np.arange(len(subset))
            np.random.shuffle(arg)
            torch.cuda.empty_cache()

        self.models['model'] = model_builder.build(self.model_config, self.data_config, self.logger)['model']

        if not isinstance(self.models['model'], torch.nn.DataParallel):
            if self._use_data_parallel():
                self.models['model'] = util.DataParallel(self.models['model'])
            self.models['model'].to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')['state_dict']
        if isinstance(self.models['model'], torch.nn.DataParallel):
            self.models['model'].module.load_state_dict(checkpoint, strict=True)
        else:
            self.models['model'].load_state_dict(checkpoint, strict=True)

        # Create a new dataloader for the updated labeled dataset
        self.dataloaders = dataloader_builder.update(
            cycle, self.dataloaders, arg, self.data_config, self.model_config, self.writer, self.save_dir)

        self.optimizer = optimizer_builder.build(
            self.train_config['optimizer'], self.models['model'].parameters(), self.logger)
        self.scheduler = scheduler_builder.build(
            self.train_config, self.optimizer, self.logger, self.train_config['num_epochs'],
            len(self.dataloaders['train']))

        if self.acquisition == 'll4al':
            self.loss_optimizer = optimizer_builder.build(
                self.train_config['optimizer'], self.models['loss_model'].parameters(), self.logger)
            self.loss_scheduler = scheduler_builder.build(
                self.train_config, self.loss_optimizer, self.logger,
                self.train_config['num_epochs'], len(self.dataloaders['train']))

