# movie_plot_generator
GPT for movie plots. Utilizes GPT2 tokenizer and Pytorch Transformer library


## Usage
Create and activate the conda environment
```
mamba env create -f env.yml
conda activate movie_gpt
```

Download the HuggingFace Model Repo
```
git clone https://huggingface.co/atancoder/movie_plot_gpt
```

Hyperparam Tuning
```
python main.py lr_grid_search
```

Train the Model
```
python main.py train_model 
```

Generate a movie plot
```
>>> python main.py gen_plot --input_prompt "Once upon a time"

Plot:  ["Once upon a time in a film publication,[2] Enoch (Pickford) is a young woman, who is given a job as a painter.
 He is sent to the hospital, where he meets a boy, a boy, and a woman (Fayah), a doctor. They decide to go to the orpha
nage, where she is in love with a philandering butler. The father is adopted by the idea of a boy named Charlie (C. Aub
rey Smith).\r\nWhen the boy dies, the stranger is brought to the hospital, where he tries to sell the boy's hand, and h
is wife, the mother's husband (Fayah) refuses to let him go. He is brought to the orphanage, and the father is sent to
the orphanage. He is brought to the orphanage, but he is not allowed to marry his daughter, but the father is adopted.
He is adopted by the older Prince, who is the mother of the Duke (Cecil), who is a very religious butler.\r\nThe boy is
 brought to the orphanage, where he finds out that the father is not a man, but the father is actually a very wealthy m
an. He tries to figure out the mother has died"]
```
