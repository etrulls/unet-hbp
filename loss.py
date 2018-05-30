import numpy as np
import torch
from IPython import embed
from torch.autograd import Variable


class OverlapLoss(object):
    def __init__(self, function='dice', smoothness=0, false_positive_factor=1):
        super().__init__()
        self.function = function
        self.smoothness = smoothness
        self.false_positive_factor = false_positive_factor
        if function not in ['dice', 'jaccard']:
            raise RuntimeError('Bad function')

    def forward(self, pred, anno):
        # Process each batch sample
        coeffs = []
        for x, y in zip(pred, anno):
            # unroll this into a vector to mask out ignored voxels
            mask = y != 2
            x = x[mask]
            y = y[mask]
            coeffs.append(1 - self.score(x, y))
        # must sum(coeffs) after
        return coeffs

    def score(self, a, b):
        bottom = a.sum() + b.sum()

        # Score is 0 if there's nothing?
        # No, let's penalize false positives
        if b.sum().data.cpu().numpy()[0] == 0:
            # print('Should never be here? (b == 0)')
            # print(self.false_positive_factor * (a.sum() / a.numel()))
            # embed()
            # return Variable(torch.from_numpy(np.array([0]))).cuda()
            return self.false_positive_factor * (a.sum() / a.numel())
        elif bottom.data.cpu().numpy()[0] == 0:
            return Variable(torch.from_numpy(np.array([0]))).cuda()
        else:
            s = 2 * ((a * b).sum() + self.smoothness) / (bottom + self.smoothness + 1e-12)
            if self.function == 'dice':
                return s
            elif self.function == 'jaccard':
                # https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient#Difference_from_Jaccard
                return s / (2 - s)
