import torch
from kosmos.model import Kosmos2

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 4096))

model = Kosmos2()
output = model(img, text)