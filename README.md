# DSPy OpenTOM

This repo contains scripts for optimizing DSPy modules for the OpenTOM Benchmark. We support Chain of Thought and a method we thought might work where we generate a "thought" about the context to aid in answering the question (spoiler -- it didn't work better than just `BootstrapFewShotWithRandomSearch`).

CLI Usage: 
```
usage: main.py [-h] [--student STUDENT] [--teacher TEACHER] [--train_size TRAIN_SIZE] [--download_dataset DOWNLOAD_DATASET]
               [--question_types [QUESTION_TYPES ...]]
               experiment_title dspy_method dspy_optimizer

Run DSPY method.

positional arguments:
  experiment_title      Title of new experiment
  dspy_method           The DSPY method to run
  dspy_optimizer        The DSPY optimizer to use

options:
  -h, --help            show this help message and exit
  --student STUDENT     The LLM to optimize prompts for
  --teacher TEACHER     Teacher LLM for optimizing prompts. Defaults to Student LLM
  --train_size TRAIN_SIZE
                        Number of training examples to use for optimization
  --download_dataset DOWNLOAD_DATASET
                        Download dataset
  --question_types [QUESTION_TYPES ...]
                        Question types. Defaults to all
```

Come chat with us in our [discord](https://discorg.gg/plasticlabs) or in the [DSPy thread](https://discord.com/channels/1161519468141355160/1214629969318252574)
