import shutil
import time

import torch
import torch.nn.functional as F

from utils import tokenize_sentences


def compute_loss(predictions, tokenized_sentences, loss_fn):
    """
    Compute loss between predictions and labels
    We have to shift the labels to the left by 1, since the idx i in the predictions is
    supposed to represent token i+1
    """
    B, T, _ = predictions.shape
    predictions = predictions.reshape(B * T, -1)[:-1]
    labels = tokenized_sentences.input_ids.reshape(B * T)[1:]

    # We only want to compute loss on non padded tokens
    loss = loss_fn(predictions, labels, reduction="none")
    padding_masks = tokenized_sentences.attention_mask.reshape(B * T)[1:]
    final_loss = loss * padding_masks
    num_non_padded_tokens = padding_masks.sum()
    return final_loss.sum() / num_non_padded_tokens


def train(
    model,
    dataloader,
    tokenizer,
    config,
    device,
    saved_model_file,
):
    start = time.time()
    size = len(dataloader.dataset)
    loss_fn = F.cross_entropy

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    model.train()
    for i in range(config["epochs"]):
        print(f"Starting epoch {i}")
        for batch_id, batch in enumerate(dataloader):
            tokenized_sentences = tokenize_sentences(
                tokenizer, batch, config["context_length"]
            ).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(tokenized_sentences.input_ids)
                loss = compute_loss(output, tokenized_sentences, loss_fn)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_id % 5 == 0:
                loss, current = loss.item(), (batch_id + 1) * len(batch)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"{((time.time() - start) / 60):.2f} minutes have elapsed")
                if saved_model_file:
                    torch.save(model.state_dict(), "tmp_model.pt")
                    shutil.move("tmp_model.pt", saved_model_file)
                    print("Saved model")


def evaluate_model(model, tokenizer, dataloader, config, device) -> float:
    model.eval()
    total_loss = 0
    loss_fn = F.cross_entropy
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            tokenized_sentences = tokenize_sentences(
                tokenizer, batch, config["context_length"]
            ).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(tokenized_sentences.input_ids)
                loss = compute_loss(output, tokenized_sentences, loss_fn)
            total_loss += loss.item()
    return total_loss / len(dataloader)
