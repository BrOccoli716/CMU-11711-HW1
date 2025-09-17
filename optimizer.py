from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
            max_grad_norm: float = None,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias, max_grad_norm=max_grad_norm)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            # TODO: Clip gradients if max_grad_norm is set
            if group['max_grad_norm'] is not None:
                # raise NotImplementedError()
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # TODO: Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lambda_ = group["weight_decay"]
                flag = group["correct_bias"]

                # TODO: Update first and second moments of the gradients
                if len(state) == 0:  # Initialize
                    state['step'] = 0
                    state['m_step'] = torch.zeros_like(p.data)
                    state['v_step'] = torch.zeros_like(p.data)
                state['step'] += 1
                g_step = p.grad.data
                state['m_step'] = beta1 * state['m_step'] + (1 - beta1) * g_step
                state['v_step'] = beta2 * state['v_step'] + (1 - beta2) * g_step * g_step

                # TODO: Bias correction
                # Please note that we are using the "efficient version" given in Algorithm 2 
                # https://arxiv.org/pdf/1711.05101
                m_step = state['m_step'] / (1 - beta1 ** state['step']) if flag else state['m_step']
                v_step = state['v_step'] / (1 - beta2 ** state['step']) if flag else state['v_step']

                # TODO: Update parameters
                p_old = p.data
                with torch.no_grad():
                    p.addcdiv_(m_step, torch.sqrt(v_step) + eps, value=-alpha)

                # TODO: Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                with torch.no_grad():
                    p.add_(p_old, alpha=-lambda_ * alpha)

        return loss
