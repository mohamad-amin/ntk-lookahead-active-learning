import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve


class MultiMeter(object):

    def __init__(self, name, logger, meters, fmt=':f'):
        self.name = name
        self.logger = logger
        self.meters = {}
        for meter_name in meters:
            self.meters[meter_name] = AverageEpochMeter(meter_name, fmt)
        self.reset()

    def reset(self):
        for key in self.meters:
            self.meters[key].reset()

    def update(self, val_dict, batch_size):
        for key in self.meters:
            self.meters[key].update(val_dict[key], batch_size)

    def print_log(self, val_dict, epoch, batch_idx, num_batches):
        return  # Todo: no bug pls
        log = ''
        for key in self.meters:
            log += key + ': {:.4f} '.format(val_dict[key])

        self.logger.info(
            '[Epoch {}] Train batch {}/{}'.format(epoch, batch_idx+1, num_batches))
        self.logger.info(log)


class AverageEpochMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, logger, fmt=':f'):
        self.name = name
        self.logger = logger
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def print_log(self, epoch, batch_idx, num_batches):
        return  # Todo: no bug pls
        self.logger.info(
            '[Epoch {}] Train batch {}/{}'.format(epoch, batch_idx+1, num_batches))
        self.logger.info('Loss: {:.4f}'.format(self.avg))


class PrecisionRecallMeter(object):
    def __init__(self, name, logger, num_classes, fmt=':f'):
        self.name = name
        self.logger = logger
        self.num_classes = num_classes
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.binary_labels = [np.array([]) for _ in range(self.num_classes)]
        self.preds = [np.array([]) for _ in range(self.num_classes)]

    def accumulate(self, labels, logits):
        preds = F.softmax(logits, dim=1).detach().cpu().numpy()
        for label, pred in zip(labels, preds):
            for cls_idx in range(self.num_classes):
                if label == cls_idx:
                    self.binary_labels[cls_idx] = np.concatenate(
                        (self.binary_labels[cls_idx], np.array([1])))
                else:
                    self.binary_labels[cls_idx] = np.concatenate(
                        (self.binary_labels[cls_idx], np.array([0])))
                self.preds[cls_idx] = np.concatenate(
                    (self.preds[cls_idx], np.array([pred[cls_idx]])))

    def compute(self):
        precision, recall, threshold = precision_recall_curve(
            self.binary_labels, self.preds)
        return precision, recall, threshold

    def get_labels_and_preds(self):
        return torch.tensor(self.binary_labels), torch.tensor(self.preds)


