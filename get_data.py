import dspy
import requests
import pickle
import json
import random
from collections import defaultdict
import pandas as pd


# this is the one that they sampled 100 existing OpenToM plots to produce "extra long" narratives
# URL = "https://raw.githubusercontent.com/SeacowX/OpenToM/main/data/opentom_long.json"
URL = "https://raw.githubusercontent.com/SeacowX/OpenToM/main/data/opentom.json"

def default_factory():
    return []

def main():
    response = requests.get(URL).json()

    df = pd.DataFrame(response)

    # Extract 'type' and 'answer' into separate columns
    df['type'] = df['question'].apply(lambda x: x['type'])
    df['answer'] = df['question'].apply(lambda x: x['answer'])

    unique_answers_by_type = df.groupby('type')['answer'].unique()

    # convert the dataset to what DSPy expects (list of Example objects)
    dataset = []

    for index, row in df.iterrows():
        context = row['narrative']
        question = row['question']['question']
        answer = row['question']['answer']
        type = row['question']['type']
        plot_info = json.dumps(row['plot_info']) # Keeping each example field as a string might be a good idea

        if "location" in type and (answer.lower().strip() != "yes" and answer.lower().strip() != "no"): # don't provide answer choices for fine grained location questions
            answer_choices = "n/a, list a specific location"
        elif "location" in type:
            answer_choices = "No, Yes"
        else:
            answer_choices = ", ".join(unique_answers_by_type[type])

        dataset.append(dspy.Example(context=context, question=question, answer=answer, type=type, plot_info=plot_info, answer_choices=answer_choices).with_inputs("context", "question", "answer_choices"))


    # split datasets by question types
    datasets = defaultdict(default_factory)

    for example in dataset:
        datasets[example.type].append(example)

    datasets.keys()
    [len(dataset) for dataset in datasets.values()]

    # create train test split
    for question_type, dataset in datasets.items():
        random.shuffle(dataset)

        datasets[question_type] = {
            "train": dataset[:int(len(dataset) * 0.8)],
            "test": dataset[int(len(dataset) * 0.8):],
        }

        print(f"Train {question_type}: {len(datasets[question_type]['train'])}")
        print(f"Test {question_type}: {len(datasets[question_type]['test'])}")

    # Serialize and save the datasets object to a file
    with open('datasets.pkl', 'wb') as file:
        pickle.dump(datasets, file)

    print("🫡 Datasets object has been saved to 'datasets.pkl' 🫡")

if __name__ == "__main__":
    main()

