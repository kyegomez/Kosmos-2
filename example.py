import torch
from nevax import Kosmos2

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = Kosmos2()
output = model(img, caption_tokens)