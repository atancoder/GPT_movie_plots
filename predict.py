import sys

import torch


def gen_tokens(model, tokenizer, input_prompt, config, device):
    # Encode input and pop off EOS
    output_tokens = tokenizer(input_prompt, return_tensors="pt").input_ids[:,:-1].to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(config["context_length"]): 
            logits = model(output_tokens)
            next_token = torch.argmax(logits[:, -1], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
            output_tokens = torch.cat([output_tokens, next_token.unsqueeze(0)], dim=1)
        sentence = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        print("Tokens: ", output_tokens)
        print("Plot: ", sentence)
