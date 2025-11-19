import torch
import torch.nn as nn

# Modelo simple: una red neuronal con 2 capas ocultas
class ModeloVentas(nn.Module):
    def __init__(self):
        super(ModeloVentas, self).__init__()
        self.fc1 = nn.Linear(1, 32)   # capa oculta 1
        self.fc2 = nn.Linear(32, 16)  # capa oculta 2
        self.fc3 = nn.Linear(16, 1)   # salida

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

modelo = ModeloVentas()

criterion = nn.MSELoss()  # error cuadrático medio
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.01)



