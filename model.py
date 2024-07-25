from typing import Dict

import torch
from torch import nn


class MovieGPTModel(nn.Module):
    def __init__(self, vocab_size: int, config: Dict, device: str):
        super().__init__()
        self.device = device
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config["channels"]
        )

        # TODO: Change to relative embeddings
        self.positional_encodings = nn.Embedding(
            num_embeddings=config["max_context_length"],
            embedding_dim=config["channels"],
        )
        self.src_mask = nn.Transformer.generate_square_subsequent_mask(
            config["context_length"]
        ).to(device)

        # We actually utilize the TransformerEncoder class but just make sure
        # to apply masking in attention
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config["channels"],
            nhead=config["num_heads"],
            dim_feedforward=config["channels"] * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer, num_layers=config["num_transformer_blocks"]
        )
        self.linear = nn.Linear(config["channels"], vocab_size)
        torch.nn.init.normal_(
            self.linear.weight, mean=0.0, std=1 / (config["channels"] ** 0.5)
        )

        # Share the same weights as the original embedding matrix
        # Works better in papers and reduces model size
        # Have embedder weight take after linear weight, b/c it's important for linear
        # weight to be initialized properly
        self.embedder.weight = self.linear.weight

    def forward(self, input):
        B, T = input.shape
        positional_embeddings = self.positional_encodings(
            torch.arange(T).to(self.device)
        )
        embeddings = self.embedder(input) + positional_embeddings
        output = self.decoder(embeddings, mask=self.src_mask[:T, :T], is_causal=True)
        return self.linear(output)
