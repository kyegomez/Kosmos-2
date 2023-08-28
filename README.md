[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# Kosmos-2
My personal implementation of Kosmos-2, Kosmos-2: Grounding Multimodal Large Language Models to the World, much simpler codebase

# Install
`pip3 install kosmos-2`

---

# Usage
```python

import torch
from kosmos.model import Kosmos2

#usage
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 4096))

model = Kosmos2()
output = model(img, text)
```

----


# Training
* [There is a file with a table of all the datasets used in the paper here](docs/datasets.md)

```python
from kosmos.train import Train


def train():
    os.environ['MASTER_ADDR'] #'localhost'
    os.environ['MASTER_PORT'] #= '9994'
    
    # # [CRITICAL] Pay attention to this when scaling to multiple GPUs and clusters
    os.environ['RANK']       #= str(0) # Number of nodes (servers)
    os.environ['WORLD_SIZE'] # = str(torch.cuda.device_count())

    dist.init_process_group(backend='nccl') #init_method="env://")
    
    Train()

if __name__ == '__main__':
    train()


```

1. Set the environment variables:
   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Directory to save the weights (e.g., `./weights`)
   - `MASTER_ADDR`: For distributed training
   - `MASTER_PORT` For master port distributed training
   - `RANK`- Number of nodes services
   - `WORLD_SIZE` Number of gpus

2. Configure the training:
   - Accelerate Config
   - Enable Deepspeed 3
   - Accelerate launch train_distributed_accelerate.py

For more information, refer to the [Training SOP](DOCs/TRAINING.md).


----



# Todo

- [ ] Multiway embeddings from zeta
- [ ] Sub layernorm
- [ ] Position aware vision language adapter, compresses image features. Singer layer cross attention module inited randomly => group of trainable embeddings as query vectors + image features from the visual encoder as keys for cross attention ops => OUTPUT: compresses visual feature sequence to a fixed lnegth of 256, 2d absolute positional encodings are integrated into the cross attentions mechanisms query key pairs => compressed feature sequence of length of 256 => fed into decoder llm

- [ ] Bounding Boxes, for any given accurate bounding box, a norm process is applied in the range [0, 1000] and transformed into a string format (Xtope, Ytople)(Xottomright, Ybottomright) -> the string is tokenized as text and does not require positional vocabulary. Detection strings and regular text strings, two special tokens <box> and </box> are added to the beginning and end of the bounding box string. + another sed of special tokens (<ref> and </ref>) is introduced.

# Citations

To cite this paper in markdown format:

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, Furu Wei. "Language Is Not All You Need: Aligning Perception with Language Models." arXiv:2302.14045 [cs.CL], 1 Mar 2023, https://doi.org/10.48550/arXiv.2302.14045.
