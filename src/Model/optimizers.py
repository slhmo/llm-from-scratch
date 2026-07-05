import torch


class CustomAdam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Custom Adam Optimizer built from scratch.
        """
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        # State initialization: tracking historical moments for every parameter tensor
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.t = 0  # Timestep counter

    def step(self):
        """Performs a single optimization step."""
        self.t += 1

        # We must disable gradient tracking during the weight update phase
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad

                # 1. Update biased first moment estimate
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

                # 2. Update biased second raw moment estimate
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                # 3. Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)

                # 4. Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                # 5. Apply adaptive step update
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()