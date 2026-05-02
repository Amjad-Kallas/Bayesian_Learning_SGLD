import numpy as np


class SGD:
    def __init__(self, a = 0.01, b = 1, gamma = 0.55):
        """Initialize SGD with polynomial step-size scheduler."""
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def lr(self):
        """Return the current SGD learning rate."""
        return self.a * (self.b + self.t) ** (-self.gamma)
    
    def step(self, theta, grad):
        """Take one SGD parameter update step using the current learning rate."""
        eps = self.lr()
        
        theta = theta + eps * grad
        
        self.t += 1

        return theta

class SGLD:
    def __init__(self, a = 0.01, b = 1, gamma = 0.55):
        """Initialize SGLD with polynomial step-size scheduler."""
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def lr(self):
        """Return the current SGLD learning rate."""
        return self.a * (self.b + self.t) ** (-self.gamma)
    
    def step(self, theta, grad):
        """Perform one SGLD update with injected Gaussian noise."""
        eps = self.lr()
        
        noise = np.random.normal(0, np.sqrt(eps), size=theta.shape)

        theta_new = theta + (0.5 * eps * grad) + noise
        
        self.t += 1

        return theta_new, eps, noise