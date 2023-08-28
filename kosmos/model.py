import logging

import torch
import torch.nn as nn
from flamingo_pytorch import PerceiverResampler
from torch.nn import Module
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from kosmos.transformer import AutoregressiveWrapper, Decoder, Transformer

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Implement classes with type hints and error handling
class Kosmos2Tokenizer:
    """
    A tokenizer class for the kosmos model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images
        tokenizer: (AutoTokenizer): The tokenizer to tokenize text 
        im_idx: (int): The Index of the "<image>" token.
        im_end_idx (int): The index of the "</image>" token.
    """

    def __init__(self):
        try:
            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )
        except Exception as e:
            logging.error(f"Failed to initialize KosmosTokenizer: {e}")
            raise

        self.im_idx, self.im_end_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>"])

    def tokenize_texts(self, texts: str):
        try:
            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            # Add image tokens to text as "<s> <image> </image> text </s>"
            image_tokens = torch.tensor([[self.im_idx, self.im_end_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], image_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            logging.error(f"Failed to tokenize texts: {e}")
            raise

    def tokenize_images(self, images):
        try:
            return self.processor(images=images, return_tensors="pt").pixel_values
        except Exception as e:
            logging.error(f"Failed to tokenize images: {e}")
            raise

    def tokenize(self, sample):
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            logging.error(f"Failed to tokenize sample: {e}")
            raise



class Kosmos2(Module):
    """
    Andromeda is a transformer-based model architecture. It initializes with 
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.
    """
    def __init__(self, 
                 num_tokens=50432, 
                 max_seq_len=8192, 
                 dim=2560, 
                 depth=32, 
                 dim_head=128, 
                 heads=24,
                 use_abs_pos_emb=False, 
                 alibi_pos_bias=True, 
                 alibi_num_heads=12, 
                 rotary_xpos=True,
                 attn_flash=True, 
                 attn_kv_heads = 2,
                 qk_norm=True, 
                 attn_qk_norm=True, 
                 attn_qk_norm_dim_scale=True, 
                 ):
        """
        Initialize the model with specified or default parameters.
        Args:
        - num_tokens: Number of tokens in the vocabulary
        - max_seq_len: Maximum sequence length
        - dim: Dimension of the model
        - depth: Depth of the model
        - dim_head: Dimension of the model head
        - heads: Number of heads
        - use_abs_pos_emb: Whether to use absolute position embedding
        - alibi_pos_bias: Alibi position bias
        - alibi_num_heads: Number of alibi heads
        - rotary_xpos: Rotary position
        - attn_flash: Attention flash
        - deepnorm: Deep normalization
        - shift_tokens: Number of tokens to shift
        - attn_one_kv_head: Attention one key/value head
        - qk_norm: Query-key normalization
        - attn_qk_norm: Attention query-key normalization
        - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
        - embedding_provider: Embedding provider module
        """
        super().__init__()

        try:
            self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
        except Exception as e:
            logging.error(f"Failed to initialize CLIP model: {e}")
            raise

        self.output_projection = nn.Linear(2048, 32002, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0, std=2048**-0.5)

        self.perceive = PerceiverResampler(
            dim=1024, 
            depth=2,
            dim_head=64,
            heads=8,
            num_latents=64,
            num_media_embeds=257
        )
        self.image_proj = nn.Linear(1024, 2048, bias=False)
        nn.init.normal_(self.image_proj.weight, mean=0, std=2048**-0.5)



        try:
            self.kosmos = Transformer(
                num_tokens=num_tokens,
                max_seq_len=max_seq_len,
                use_abs_pos_emb=use_abs_pos_emb,
                attn_layers=Decoder(
                    dim=dim,
                    depth=depth,
                    dim_head=dim_head,
                    heads=heads,
                    cross_attend=True,
                    alibi_pos_bias=alibi_pos_bias,
                    alibi_num_heads=alibi_num_heads,
                    rotary_xpos=rotary_xpos,
                    attn_flash=attn_flash,
                    attn_kv_heads=attn_kv_heads,
                    qk_norm=qk_norm,
                    attn_qk_norm=attn_qk_norm,
                    attn_qk_norm_dim_scale=attn_qk_norm_dim_scale
                )
            )

            self.decoder = AutoregressiveWrapper(self.kosmos)

        except Exception as e:
            print("Failed to initialize Andromeda: ", e)
            raise

    def forward(self, text, images, **kwargs):
        """
        Forward pass through the model. It expects the input text_tokens.
        Args:
        - text_tokens: Input tokens
        - kwargs: Other arguments
        Returns:
        - output from the decoder
        """
        try:
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)

            model_input = self.decoder.forward(text, context=images)[0]
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            print("Failed in forward method: ", e)
            raise