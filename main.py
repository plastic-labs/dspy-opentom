# run with python main.py cot

import pickle
import time
import argparse
from opentom_evaluator import OpenToMEvaluatorDspy
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.evaluate.evaluate import Evaluate
from cot import CoTSimplifiedBaleen
from get_data import default_factory
from collections import defaultdict



EVAL_QUESTION_TYPES = ["attitude", "multihop-fo", "multihop-so", "location-fo-coarse", "location-fo-fine", "location-so-coarse", "location-so-fine"]

def dump_state(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)

def main(dspy_method):

    # read in the datasets pickle object
    with open('datasets.pkl', 'rb') as file:
        datasets = pickle.load(file)

    if dspy_method == 'cot':
        modules = {}
        # define modules for each question type
        for question_type in EVAL_QUESTION_TYPES:
            print(f"TYPE: {question_type}")
            evaluator = OpenToMEvaluatorDspy(model_name="(training set) complied baleen")
            optimizer = BootstrapFewShotWithRandomSearch(metric=evaluator.dspy_metric, num_candidate_programs=50, num_threads=1)
            compiled_baleen = optimizer.compile(CoTSimplifiedBaleen(), trainset=datasets[question_type]["train"][:10])

            modules[question_type] = compiled_baleen
            time.sleep(60)

        uncompiled_baleen = CoTSimplifiedBaleen()

        print("Macro Averaged F1 Scores")
        for question_type in EVAL_QUESTION_TYPES:
            test = datasets[question_type]["test"][:25]
            compiled_baleen = modules[question_type]

            # Set up the `evaluate_on_hotpotqa` function.
            evaluate_on_opentom = Evaluate(devset=test, num_threads=1, display_progress=True, display_table=10)

            uncompiled_baleen_evaluator = OpenToMEvaluatorDspy(model_name='uncompiled_baleen')
            evaluate_on_opentom(uncompiled_baleen, metric=uncompiled_baleen_evaluator.dspy_metric, display=False)
            uncompiled_baleen_evaluator.print_f1_results()

            compiled_baleen_evaluator = OpenToMEvaluatorDspy(model_name='compiled_baleen')
            evaluate_on_opentom(compiled_baleen, metric=compiled_baleen_evaluator.dspy_metric, display=False)
            compiled_baleen_evaluator.print_f1_results()

        dump_state(modules, 'cot_modules.pkl')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run DSPY method.')
    parser.add_argument('dspy_method', type=str, help='The DSPY method to run')
    args = parser.parse_args()

    main(args.dspy_method)