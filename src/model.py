# model.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, activation_name, hidden_size=512, num_layers=4):
        super(MLP, self).__init__()
        
        # 1. Выбор функции активации
        if activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
            self.init_type = 'standard' # Pytorch default (uniform)
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
            self.init_type = 'xavier'    # Инициализация Глорота
        elif activation_name == 'relu':
            self.activation = nn.ReLU()
            self.init_type = 'kaiming'   # Инициализация He
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        # 2. Определение слоев
        layers = [nn.Flatten()]
        in_features = 28 * 28 # MNIST/Fashion-MNIST
        
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self.activation)
            in_features = hidden_size
        
        # Выходной слой (10 классов)
        layers.append(nn.Linear(hidden_size, 10))
        
        self.net = nn.Sequential(*layers)
        
        # 3. Применение выбранной инициализации
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Glorot/Xavier для tanh (использует коэффициент 1.0)
            if self.init_type == 'xavier':
                # k: Fan-in/fan-out ratio для Glorot Uniform
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('tanh'))
            # Kaiming/He для ReLU (использует коэффициент 2.0)
            elif self.init_type == 'kaiming':
                # k: Fan-in/fan-out ratio для He/Kaiming Uniform
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            # Стандартная инициализация (для Sigmoid)
            else:
                # Стандартная инициализация PyTorch (Uniform)
                pass 
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)

# Конец model.py