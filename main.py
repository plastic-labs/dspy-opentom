# run with python main.py cot

import pickle
import time
import argparse
from typing import Optional
from opentom_evaluator import OpenToMEvaluatorDspy
import dspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, SignatureOptimizer
from dspy.evaluate.evaluate import Evaluate
from cot import CoTSimplifiedBaleen
from get_data import default_factory, load_dataset
from collections import defaultdict
from dotenv import load_dotenv
import neptune
import numpy as np

load_dotenv()

# initialize neptune
run = neptune.init_run(
    project="dspy-opentom/dspy-evaluation",
    capture_hardware_metrics=False,
    capture_stderr=True,
    capture_stdout=True,
    capture_traceback=True,
)

EVAL_QUESTION_TYPES = [
    "attitude",
    "multihop-fo",
    "multihop-so",
    "location-fo-coarse",
    "location-fo-fine",
    "location-so-coarse",
    "location-so-fine",
]


def dump_state(data, filename):
    with open(filename, "wb") as file:
        pickle.dump(data, file)


def main(dspy_method, dspy_optimizer, download_dataset, question_types, teacher_lm, train_size):
    # load dataset
    if download_dataset:
        load_dataset()

    # read in the datasets pickle object
    with open("datasets.pkl", "rb") as file:
        datasets = pickle.load(file)

    if dspy_method == "cot":
        modules = {}
        # define modules for each question type
        for question_type in question_types:
            print(f"TYPE: {question_type}")
            evaluator = OpenToMEvaluatorDspy(model_name="(training set) complied baleen")

            if dspy_optimizer == "bootstrap_fewshot_with_random_search":
                optimizer = BootstrapFewShotWithRandomSearch(
                    metric=evaluator.dspy_metric,
                    num_candidate_programs=25,
                    num_threads=1,
                    teacher_settings=dict(lm=teacher_lm),
                )
                compiled_baleen = optimizer.compile(
                    CoTSimplifiedBaleen(), trainset=datasets[question_type]["train"][:train_size]
                )
            elif dspy_optimizer == "signature_optimizer":
                optimizer = SignatureOptimizer(
                    metric=evaluator.dspy_metric,
                    breadth=10,
                    depth=3,
                    init_temperature=1.4,
                    verbose=True,
                    track_stats=True,
                    prompt_model=teacher_lm,
                )
                eval_kwargs = dict(num_threads=1, display_progress=True, display_table=0)
                compiled_baleen = optimizer.compile(
                    CoTSimplifiedBaleen(),
                    devset=datasets[question_type]["train"][:train_size],
                    eval_kwargs=eval_kwargs,
                )
            else:
                raise Exception(f"Invalid dspy optimizer type: {dspy_optimizer}")

            modules[question_type] = compiled_baleen
            time.sleep(10)

        uncompiled_baleen = CoTSimplifiedBaleen()

        print("Beginning Evaluation")
        for question_type in question_types:
            compiled_baleen = modules[question_type]

            # Evaluation Procedure: Calculate the F1 Score for a randomly drawn batch of 50 questions 5 times and average the F1 Scores
            batch_size = 50
            num_batches = 5

            assert len(datasets[question_type]["test"]) >= batch_size * num_batches
            test = datasets[question_type]["test"][: batch_size * num_batches]
            test_sets = [test[i : i + batch_size] for i in range(num_batches)]

            uncompiled_f1_scores = []
            compiled_f1_scores = []

            for test in test_sets:
                # Set up the `evaluate_on_hotpotqa` function.
                evaluate_on_opentom = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=0)

                uncompiled_baleen_evaluator = OpenToMEvaluatorDspy(model_name="uncompiled_baleen")
                evaluate_on_opentom(uncompiled_baleen, metric=uncompiled_baleen_evaluator.dspy_metric, display=True)
                uncompiled_f1_scores.append(uncompiled_baleen_evaluator.f1_score()[question_type]["macro_averaged"])

                compiled_baleen_evaluator = OpenToMEvaluatorDspy(model_name="compiled_baleen")
                evaluate_on_opentom(compiled_baleen, metric=compiled_baleen_evaluator.dspy_metric, display=True)
                compiled_f1_scores.append(compiled_baleen_evaluator.f1_score()[question_type]["macro_averaged"])

            # overall f1 scores
            uncompiled_mean_f1 = np.mean(uncompiled_f1_scores)
            uncompiled_std_f1 = np.std(uncompiled_f1_scores)

            compiled_mean_f1 = np.mean(compiled_f1_scores)
            compiled_std_f1 = np.std(compiled_f1_scores)

            run[f"evaluation/{question_type}/uncompiled/mean_macro_averaged_f1"] = uncompiled_mean_f1
            run[f"evaluation/{question_type}/uncompiled/mean_macro_averaged_f1"] = uncompiled_std_f1
            run[f"evaluation/{question_type}/compiled/mean_macro_averaged_f1"] = compiled_mean_f1
            run[f"evaluation/{question_type}/compiled/mean_macro_averaged_f1"] = compiled_std_f1

            print(
                f"Mean Macro Averaged F1 Scores (± std dev.) - {question_type} - Aggregated from {num_batches} batches of {batch_size} questions"
            )
            print(f"uncompiled: {uncompiled_mean_f1:.3f} ± {uncompiled_std_f1:.3}")
            print(f"compiled: {compiled_mean_f1:.3} ± {compiled_std_f1:.3}")

        dump_state(modules, "cot_modules.pkl")
        run["cot_modules"].upload("cot_modules.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DSPY method.")

    # dspy arguments
    parser.add_argument("experiment_title", type=str, help="Title of new experiment")
    parser.add_argument("dspy_method", type=str, help="The DSPY method to run")
    parser.add_argument("dspy_optimizer", type=str, help="The DSPY optimizer to use")
    parser.add_argument("--student", default="gpt-3.5-turbo", type=str, help="The LLM to optimize prompts for")
    parser.add_argument("--teacher", default=None, type=str, help="Teacher LLM for optimizing prompts. Defaults to Student LLM")
    parser.add_argument("--train_size", default=50, type=int, help="Number of training examples to use for optimization")
    parser.add_argument("--download_dataset", default=True, type=bool, help="Download dataset")
    parser.add_argument("--question_types", default=EVAL_QUESTION_TYPES, nargs="*", help="Question types. Defaults to all")

    args = parser.parse_args()

    # setup LLMs
    student_lm = dspy.OpenAI(model=args.student, max_tokens=1000)
    args.teacher = args.student if args.teacher is None else args.teacher
    teacher_lm = dspy.OpenAI(model=args.teacher, max_tokens=1000)
    dspy.settings.configure(lm=student_lm)

    # validate question types
    question_types = args.question_types
    assert all([question_type in EVAL_QUESTION_TYPES for question_type in question_types])
    args.question_types = ", ".join(question_types)  # turn list into string for neptune logging

    # log run parameters
    run["parameters"] = args
    run["sys/name"] = args.experiment_title

    main(args.dspy_method, args.dspy_optimizer, args.download_dataset, question_types, teacher_lm, args.train_size)
