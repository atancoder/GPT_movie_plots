import os

import click
import torch
import yaml
from transformers import GPT2Tokenizer

from model import MovieGPTModel
from predict import gen_tokens
from train import evaluate_model, train
from utils import get_dataloader

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {DEVICE} device")
torch.manual_seed(1337)
torch.set_float32_matmul_precision("high")


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
GPT2Tokenizer.pad_token = GPT2Tokenizer.eos_token
GPT2Tokenizer.pad_token_id = GPT2Tokenizer.eos_token_id


@click.group()
def cli():
    pass


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# LR grid search
LEARNING_RATE_SEARCH = [10**r for r in range(-5, 0)]


def get_model(tokenizer, config, saved_model_file):
    model = MovieGPTModel(tokenizer.vocab_size, config, DEVICE)
    print(f"Model has {sum(p.numel() for p in model.parameters()) / 1e6}M params")
    model = torch.compile(model)
    if os.path.exists(saved_model_file):
        print(f"Loading existing model: {saved_model_file}")
        model.load_state_dict(
            torch.load(
                saved_model_file, map_location=torch.device(DEVICE), weights_only=True
            )
        )
    return model.to(DEVICE)


@click.command(name="lr_grid_search")
@click.option("--train_data", default="movie_plots.csv.gz")
@click.option("--model_config", default="model_config.yaml")
def lr_grid_search(train_data, model_config):
    config = load_config(model_config)
    config["epochs"] = 10
    dataloader = get_dataloader(train_data, config["batch_size"], max_data_size=100)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config["max_context_length"]
    scores = {}
    for lr in LEARNING_RATE_SEARCH:
        print(f"Evaluating LR {lr}")
        config["learning_rate"] = lr
        model = get_model(tokenizer, config, None)
        train(
            model,
            dataloader,
            tokenizer,
            config,
            DEVICE,
            None,
        )
        scores[lr] = evaluate_model(model, tokenizer, dataloader, config, DEVICE)
    print(f"LR Scores: {scores}")
    best_lr = min(scores.items(), key=lambda x: x[1])
    print(f"Best LR: {best_lr}")


@click.command(name="train_model")
@click.option(
    "--saved_model_file", "saved_model_file", default="movie_plot_gpt/model.pt"
)
@click.option("--train_data", default="movie_plots.csv.gz")
@click.option("--model_config", default="model_config.yaml")
def train_model(saved_model_file, train_data, model_config):
    config = load_config(model_config)
    dataloader = get_dataloader(train_data, config["batch_size"], max_data_size=10000)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config["max_context_length"]
    model = get_model(tokenizer, config, saved_model_file)
    train(
        model,
        dataloader,
        tokenizer,
        config,
        DEVICE,
        saved_model_file,
    )


@click.command(name="gen_plot")
@click.option("--model", "saved_model_file", default="movie_plot_gpt/model.pt")
@click.option("--input_prompt", default="")
@click.option("--model_config", default="model_config.yaml")
def gen_plot(
    saved_model_file: str,
    input_prompt: str,
    model_config,
):
    config = load_config(model_config)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = config["max_context_length"]
    if not os.path.exists(saved_model_file):
        raise Exception("Must have a pretrained model")
    model = get_model(tokenizer, config, saved_model_file)
    gen_tokens(
        model,
        tokenizer,
        input_prompt,
        config,
        DEVICE,
    )


cli.add_command(train_model)
cli.add_command(gen_plot)
cli.add_command(lr_grid_search)

if __name__ == "__main__":
    cli()
