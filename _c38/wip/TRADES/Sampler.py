from abc import ABC, abstractmethod
import torch
import numpy as np
import constants as cst

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device=None):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to (defaults to cst.DEVICE;
                       pass the batch's device for DDP correctness).
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        if device is None:
            device = cst.DEVICE
        w = self.weights()
        # Robustness: a single NaN/inf loss must not be able to poison the
        # importance distribution and crash np.random.choice (which rejects
        # probabilities containing NaN). Sanitize then fall back to uniform.
        w = np.nan_to_num(np.asarray(w, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if not np.isfinite(s) or s <= 0:
            w = np.ones(len(w), dtype=np.float64)
            s = w.sum()
        p = w / s
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device, non_blocking=True)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device, non_blocking=True)
        return indices, weights

    
class LossSecondMomentResampler(ScheduleSampler):
    def __init__(self, num_diffusionsteps, history_per_term=10, uniform_prob=0.001):
        self.num_diffusionsteps = num_diffusionsteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [num_diffusionsteps, history_per_term], dtype=np.float32
        )
        self._loss_counts = np.zeros([num_diffusionsteps], dtype=np.int32)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_diffusionsteps], dtype=np.float32)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_losses(self, ts, losses):
        for i in range(len(losses)):
            t = ts[i].item()
            loss = losses[i].item()
            # Never store a non-finite loss: it would permanently poison this
            # timestep's history and, through weights(), the sampling dist.
            if not np.isfinite(loss):
                continue
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, int(self._loss_counts[t])] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()