# DSPy OpenTOM

This repo contains scripts for optimizing DSPy modules for the OpenTOM Benchmark.

CLI Usage: ```
usage: main.py [-h] [--student STUDENT] [--teacher TEACHER] [--train_size TRAIN_SIZE] [--download_dataset DOWNLOAD_DATASET] [--question_types [QUESTION_TYPES ...]]
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
