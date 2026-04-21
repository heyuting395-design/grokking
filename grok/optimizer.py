import torch
import math

class CustomAdamW(torch.optim.Optimizer):
    """
    OpenAI Original CustomAdamW implementation.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        noise_factor=0.0,
        weight_decay_form="to_zero",
    ):
        if not 0.0 <= lr: raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps: raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0: raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0: raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, noise_factor=noise_factor, weight_decay_form=weight_decay_form)
        super(CustomAdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CustomAdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                grad = p.grad
                if group["weight_decay"] > 0 and group["weight_decay_form"] == "honest":
                    grad = grad + group["weight_decay"] * p.detach()

                if grad.is_sparse: raise RuntimeError("Adam does not support sparse gradients")
                
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["weight_decay_form"] == "to_init":
                        state["init"] = p.detach().clone()

                if group["weight_decay"] > 0:
                    if group["weight_decay_form"] == "to_zero":
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    elif group["weight_decay_form"] == "to_init":
                        p.add_((state["init"] - p) * (group["lr"] * group["weight_decay"]))
                    elif group["weight_decay_form"] == "jiggle":
                        p.mul_(torch.exp(torch.randn(1).to(p.device) * (group["lr"] * group["weight_decay"])))

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state["step"])).add_(group["eps"])
                step_size = group["lr"] / (1 - beta1 ** state["step"])
                
                upd = exp_avg / denom
                if group["noise_factor"] > 0:
                    upd += torch.randn_like(upd) * group["noise_factor"]
                p.add_(-step_size * upd)
        return loss