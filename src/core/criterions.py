import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from torch.nn import MSELoss


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1., temp=1., reduction='mean', eps=1e-6):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps

    def forward(self, preds, labels):
        preds = preds / self.temp
        if self.gamma >= 1.:
            loss = F.cross_entropy(
                preds, labels, weight=self.weight, reduction=self.reduction)
        else:
            log_prob = preds - torch.logsumexp(preds, dim=1, keepdim=True)
            log_prob = log_prob * self.gamma
            loss = F.nll_loss(
                log_prob, labels, weight=self.weight, reduction=self.reduction)

        return loss


class FocalLoss(_Loss):
    def __init__(self, weight=None, alpha=1., gamma=1., reduction='mean'):
        super(_Loss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        log_prob = F.log_softmax(preds, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            (self.alpha * (1 - prob) ** self.gamma) * log_prob, labels,
            weight=self.weight, reduction = self.reduction)
        losses = {'loss': loss}
        return losses

class LossPredLoss(_Loss):
    def __init__(self, margin, reduction='mean'):
        super(_Loss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, input, target):
        assert len(input) % 2 == 0, 'the batch size is not even.'
        assert input.shape == input.flip(0).shape

        input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
        target = (target - target.flip(0))[:len(target)//2]
        target = target.detach()

        one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors

        if self.reduction == 'mean':
            loss = torch.sum(torch.clamp(self.margin - one * input, min=0))
            loss = loss / input.size(0) # Note that the size of input is already halved
        elif self.reduction == 'none':
            loss = torch.clamp(self.margin - one * input, min=0)
        else:
            NotImplementedError()

        return loss


class CustomCriterion(_Loss):
    def __init__(self, name, num_classes):
        super(_Loss, self).__init__()
        self.name = name
        self.num_classes = num_classes

        if 'l2' == self.name:
            self.criterion = MSELoss()
        elif 'll4al_l2' == self.name:
            #self.criterion = CrossEntropyLoss(reduction='none')
            self.criterion = MSELoss(reduction='none')
            self.loss_criterion = LossPredLoss(margin=1.0)
            self.weight = 1.0
        elif 'll4al' == self.name:
            self.criterion = CrossEntropyLoss(reduction='none')
            self.loss_criterion = LossPredLoss(margin=1.0)
            self.weight = 1.0
        else:
            self.criterion = CrossEntropyLoss()


    def forward(self, output_dict):
        logits = output_dict['logits']
        labels = output_dict['labels']
        if 'l2' in self.name:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
            if self.name == 'softmax_l2':
                logits = F.softmax(logits, dim=1)

        loss = self.criterion(logits, labels)
        if 'pred_loss' in output_dict:
            if len(loss.shape) == 2:
                loss = torch.sum(loss, dim=1)
            pred_loss = output_dict['pred_loss']
            if len(pred_loss) % 2 != 0:
                pred_loss, loss = pred_loss[:-1], loss[:-1]
            loss_pred_loss = self.loss_criterion(pred_loss, loss)
            actual_loss = torch.sum(loss) / loss.size(0)
            loss = actual_loss + self.weight * loss_pred_loss

        losses = {'loss': loss}
        return losses
