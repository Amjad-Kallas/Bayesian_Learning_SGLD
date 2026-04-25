import numpy as np


class SGD:
    def __init__(self, a = 0.01, b = 1, gamma = 0.55):
        """
        Polynomial Learning Rate (as in the paper):
        eps_t = a * (b + t)^(-gamma)
        """
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def lr(self):
        return self.a * (self.b + self.t) ** (-self.gamma)
    
    def step(self, theta, grad):
        eps = self.lr()
        
        theta = theta + eps * grad
        
        self.t += 1

        return theta

class SGLD:
    def __init__(self, a = 0.01, b = 1, gamma = 0.55):
        """
        Polynomial Learning Rate (as in the paper):
        eps_t = a * (b + t)^(-gamma)
        """
        self.a = a
        self.b = b
        self.gamma = gamma
        self.t = 0
        
    def lr(self):
        return self.a * (self.b + self.t) ** (-self.gamma)
    
    def step(self, theta, grad):
        eps = self.lr()
        
        noise = np.random.normal(0, np.sqrt(eps), size=theta.shape)

        theta_new = theta + (0.5 * eps * grad) + noise
        
        self.t += 1

        return theta_new, eps, noise