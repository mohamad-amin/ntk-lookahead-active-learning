import torch

class Evaluator(object):

    def __init__(self, logger):
        self.logger = logger

    def reset(self):
        pass

    def update(self, output_dict):
        pass

    def print_log(self, epoch, num_steps):
        pass


class AccEvaluator(Evaluator):

    def __init__(self, logger):
        self.logger = logger
        self.reset()

    def reset(self):
        self.acc = 0
        self.num_total = 0.
        self.num_correct = 0.

    def update(self, output_dict):
        logits = output_dict['logits']
        labels = output_dict['labels']
        accuracy = compute_accuracy(logits, labels)
        # accuracy = topk_accuracy(logits, labels, topk=(1,))[0]

        batch_size = logits.shape[0]
        self.num_correct += accuracy * batch_size
        self.num_total += batch_size

    def compute(self):
        self.acc = self.num_correct / float(self.num_total)
        return self.acc


def compute_accuracy(outputs, labels):
    return (outputs.argmax(axis=1) == labels).sum() / len(labels)


def topk_accuracy(outputs, labels, topk=(1,)):
    """Computes the accuracy for the top k predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = torch.topk(outputs, k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        topk_accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=False)
            topk_accuracies.append(correct_k.mul_(1.0 / batch_size).item())
        return topk_accuracies

