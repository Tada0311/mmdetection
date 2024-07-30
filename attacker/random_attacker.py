import torch


class RandomAttacker():
    def __init__(self, data_batch):
        self.data_batch = data_batch
        self.origin_tensor = data_batch['inputs'][0]
        self.eps = 8
        self.calib = int(self.eps / 8)


    def initialize_noise(self):
        self.noise = torch.randint(0, 2, self.origin_tensor.shape, dtype=torch.int8) * (2 * self.eps) - self.eps
        self.noisy_tensor = self.origin_tensor.to(torch.int16) + self.noise
        self.noisy_tensor = torch.clamp(self.noisy_tensor, 0, 255).to(torch.uint8)

        self.data_batch['inputs'][0] = self.noisy_tensor

        return self.data_batch
    
    def search_noise(self):
        adjustment = torch.randint(0, 2, self.noisy_tensor.shape, dtype=torch.int8) * (2 * self.calib) - self.calib
        self.noise = self.noise.to(torch.int16) + adjustment
        self.noise = torch.clamp(self.noise, -self.eps, self.eps)
        
        self.noisy_tensor = self.origin_tensor.to(torch.int16) + self.noise
        self.noisy_tensor = torch.clamp(self.noisy_tensor, 0, 255).to(torch.uint8)

        self.data_batch['inputs'][0] = self.noisy_tensor

        return self.data_batch
