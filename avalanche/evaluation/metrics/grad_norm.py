import torch
import copy
from typing import TYPE_CHECKING, Dict, List
from torch import Tensor
from collections import defaultdict

from avalanche.evaluation import PluginMetric, GenericPluginMetric, Metric
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metrics.mean import Mean
from avalanche.evaluation.metric_utils import get_metric_name

if TYPE_CHECKING:
    from avalanche.training.strategies import BaseStrategy

def _extract_gradient(parameters):
    return [p.grad for p in parameters if p.grad is not None]

def _compute_l2_norm(gradient_vector):
    norm = 0.
    total = 0.
    for g in gradient_vector:
        norm += torch.sum(torch.square(g))
        total += torch.numel(g)
    norm /= total
    norm = torch.sqrt(norm)
    return float(norm.cpu().numpy())


class GradNormMetric(Metric[float]):
    """
    Standalone gradient norm metric, keeps a dictionnary of norm (one per task)

    Warning: This will significantly slow down the training since it has 
    to compute per-task gradients one more time
    """

    def __init__(self):
        """
        Creates an instance of the Norm metric
        """
        self._mean_norm = defaultdict(Mean)

    def update(self, task_gradient: List[Tensor], 
               task_total: Tensor, tid: int) -> None:
        """
        Update the running norm

        :return: None.
        """
        task_norm = _compute_l2_norm(task_gradient)
        self._mean_norm[tid].update(task_norm, task_total)
            

    def result(self, task_label=None) -> float:
        """
        Retrieves the norm result

        :return: The average norm per task
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            return {k: v.result() for k, v in self._mean_norm.items()}
        else:
            return {task_label: self._mean_norm[task_label].result()}

    def reset(self, task_label=None) -> None:
        """
        Resets the metric.

        :return: None.
        """
        assert(task_label is None or isinstance(task_label, int))
        if task_label is None:
            self._mean_norm = defaultdict(Mean)
        else:
            self._mean_norm[task_label].reset()


class GradNormPluginMetric(PluginMetric[float]):
    """
    Compute the gradient norm at each backward pass
    """

    def __init__(self):
        """
        Compute the gradient norm (per task) at each backward pass
        """
        super().__init__()
        self._norm = GradNormMetric()

    def update(self, strategy) -> Tensor:
        """
        Update the weight checkpoint at the current experience.

        :param weights: the weight tensor at current experience
        :return: None.
        """
        for tid in torch.unique(strategy.mb_task_id):
            task_output = strategy.mb_output[strategy.mb_task_id == tid]
            task_targets = strategy.mb_y[strategy.mb_task_id == tid]
            task_loss = strategy._criterion(task_output, task_targets)
            task_gradient = torch.autograd.grad(task_loss, 
                                                strategy.model.parameters(),
                                                retain_graph=True)
            task_total = len(task_output)
            self._norm.update(task_gradient, task_total, int(tid))


    def result(self) -> Tensor:
        """
        Retrieves the current grad norm

        :return: L2 grad norm
        """
        return self._norm.result()
    
    def reset(self) -> None:
        """
        This metric is resetted at every computation

        :return: None.
        """
        self._norm.reset()

    def _package_result(self, strategy: 'BaseStrategy') -> 'MetricResult':
        metric_value = self.result()
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=False, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        else:
            metric_name = get_metric_name(self, strategy,
                                          add_experience=False,
                                          add_task=True)
            return [MetricValue(self, metric_name, metric_value,
                                plot_x_position)]

    def before_backward(self, strategy, **kwargs):
        self.reset()
        self.update(strategy)
        return self._package_result(strategy) 
        
    def __str__(self):
        return 'GradNormMetric'


__all__ = ['GradNormPluginMetric']
