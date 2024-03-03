# taken from https://github.com/seacowx/OpenToM/blob/main/src/evaluate/opentom_evaluator.py
# modified for usability

from collections import defaultdict
import json
import traceback


class OpenToMEvaluatorDspy:

    def __init__(self, model_name="") -> None:
        self.true_positives = defaultdict(lambda: 0)
        self.false_positives = defaultdict(lambda: 0)
        self.false_negatives = defaultdict(lambda: 0)
        self.model_name = model_name

    def dspy_metric(self, example, pred_answer, trace=None):
        type = example.type

        eval_result = self.check_answer(
            example, pred_answer.answer
        )
        if (
            eval_result == None
        ):  # Hm what is the correct value to return as a dspy metric when there's an invalid example?
            return None
        gt, pred = eval_result  # ground truth answer class, predicted answer class

        # store positive/negative results by class so we can calculate the f1 scores later
        if gt == pred:
            self.true_positives[f"{type}_{pred}"] += 1
        else:
            self.false_positives[f"{type}_{pred}"] += 1
            self.false_negatives[f"{type}_{gt}"] += 1

        # print("done", example.type, gt, pred, example.answer, pred_answer.answer)

        return gt == pred

    # this method was added to make dspy evaluation easier
    def check_answer(
        self,
        example,
        pred_answer,
        cot_flag=False,
        perspective="all",
    ):
        mover, affected_char, eoi, original_place, move_to_place = json.loads(
            example.plot_info
        ).values()

        cur_question_type = example.type
        question_content = example.question

        gt_answer = example.answer.strip()
        pred_answer = pred_answer.strip()

        # NOTE: evaluate based on the character
        if perspective == "observer":
            if mover in question_content and affected_char not in question_content:
                return None

            if mover in question_content and affected_char in question_content:
                question_tokens = (
                    question_content.replace("'s", "").replace(",", "").split()
                )

                mover_idx = question_tokens.index(mover)
                affected_char_idx = question_tokens.index(affected_char)

                if mover_idx < affected_char_idx:
                    return None

        elif perspective == "mover":
            if mover not in question_content and affected_char in question_content:
                return None

            if mover in question_content and affected_char in question_content:
                question_tokens = (
                    question_content.replace("'s", "").replace(",", "").split()
                )

                mover_idx = question_tokens.index(mover)
                affected_char_idx = question_tokens.index(affected_char)

                if mover_idx > affected_char_idx:
                    return None

        if cot_flag:
            pred_answer = self.parse_cot_answer(pred_answer)

        if cur_question_type == "location-fo-coarse":
            gt, pred = self.check_answer_for_cg_location(pred_answer, gt_answer)
            return gt, pred
        
        elif cur_question_type == "location-fo-fine":
            gt, pred = self.check_answer_for_fg_location(
                pred_answer, gt_answer, original_place, move_to_place
            )
            return gt, pred

        elif cur_question_type == "location-so-coarse":
            gt, pred = self.check_answer_for_cg_location(pred_answer, gt_answer)
            return gt, pred
        
        elif cur_question_type == "location-so-fine":
            gt, pred = self.check_answer_for_fg_location(
                pred_answer, gt_answer, original_place, move_to_place
            )
            return gt, pred

        elif cur_question_type == "multihop-fo":
            if "fullness" in question_content:
                gt, pred = self.check_fullness_answer(pred_answer, gt_answer)
                return gt, pred

            elif "accessibility" in question_content:
                if "|" in gt_answer:
                    gt_answer = "equally accessible"

                if isinstance(gt_answer, list):
                    gt_answer = [ele for ele in gt_answer if ele != "corrupted"]
                    assert len(gt_answer) == 1
                    gt_answer = gt_answer[0]

                gt, pred = self.check_accessibility_answer(pred_answer, gt_answer)
                return gt, pred

        elif cur_question_type == "multihop-so":
            if "fullness" in question_content:
                gt, pred = self.check_fullness_answer(pred_answer, gt_answer)
                return gt, pred

            elif "accessibility" in question_content:
                if "|" in gt_answer:
                    gt_answer = "equally accessible"

                if isinstance(gt_answer, list):
                    gt_answer = [ele for ele in gt_answer if ele != "corrupted"]
                    assert len(gt_answer) == 1
                    gt_answer = gt_answer[0]

                gt, pred = self.check_accessibility_answer(pred_answer, gt_answer)
                return gt, pred

        elif cur_question_type == "attitude":
            gt, pred = self.check_attitude_answer(pred_answer, gt_answer)
            return gt, pred

    def f1_score(self):
        true_positives = self.true_positives
        false_positives = self.false_positives
        false_negatives = self.false_negatives
        f1_scores = defaultdict(lambda: {"by_class": {}})

        for _class in (
            true_positives.keys() | false_positives.keys() | false_negatives.keys()
        ):
            question_type, _ = _class.split("_")
            class_true_positives = true_positives[_class]
            class_false_positives = false_positives[_class]
            class_false_negatives = false_negatives[_class]
            class_precision = (
                class_true_positives / (class_true_positives + class_false_positives)
                if class_true_positives > 0.0
                else 0.0
            )  # avoid dividing by zero
            class_recall = (
                class_true_positives / (class_true_positives + class_false_negatives)
                if class_true_positives > 0.0
                else 0.0
            )
            class_f1_score = (
                (2 * class_precision * class_recall) / (class_precision + class_recall)
                if class_precision > 0.0 or class_recall > 0.0
                else 0.0
            )
            f1_scores[question_type]["by_class"][_class] = class_f1_score

        for question_type, type_f1_scores in f1_scores.items():
            type_f1_scores = type_f1_scores["by_class"]
            macro_averaged_f1_score = sum(list(type_f1_scores.values())) / len(
                type_f1_scores
            )
            f1_scores[question_type]["macro_averaged"] = macro_averaged_f1_score

        return f1_scores

    # pretty print macro averaged f1 scores for each question type
    def print_f1_results(self, round_decimal=2, print_header=False):
        f1_scores = self.f1_score()
        if print_header:
            print("Macro Averaged F1 Scores by question type")

        print(self.model_name, end=" - ")
        for question_type, type_f1_scores in f1_scores.items():
            print(
                f'{question_type}: {round(type_f1_scores["macro_averaged"], ndigits=round_decimal+2) * 100}',
                end="\t",
            )
        print()

    @staticmethod
    def remove_determinant(word: str) -> str:
        determinants = ["a", "an", "the"]
        for det in determinants:
            if word.startswith(det):
                return word[len(det) :].strip()
        return word

    @staticmethod
    def compute_lexical_overlap(pred: str, location: str) -> float:
        pred = pred.lower().replace("_", " ").replace("'s", "")
        location = location.lower().replace("_", " ").replace("'s", "")
        score = 0
        pred = pred.replace(".", "").split()
        location = location.split()
        visited_word = []

        for word in pred:
            if word in location and word not in visited_word:
                score += 1
                visited_word.append(word)

        return score / len(location)

    def parse_cot_answer(self, answer: str) -> str:
        # cot typically generate answer in the last sentence or paragraph
        if "\n" in answer:
            answer = answer.split("\n")[-1]
        else:
            answer = answer.split("Therefore")[-1]
        return answer

    def check_answer_for_fg_location(
        self, prediction: str, answer: str, original_place: str, move_to_place: str
    ) -> list:

        # truncate prediction as some of them contain explanations
        answer = self.remove_determinant(answer).lower()
        original_place = self.remove_determinant(original_place).lower()
        move_to_place = self.remove_determinant(move_to_place).lower()
        gt_label, pred_label = None, None
        original_place_score = self.compute_lexical_overlap(prediction, original_place)
        move_to_place_score = self.compute_lexical_overlap(prediction, move_to_place)

        if original_place_score == move_to_place_score:
            pred_label = 3
        if original_place_score > move_to_place_score:
            pred_label = 1
        elif original_place_score < move_to_place_score:
            pred_label = 2

        if original_place == answer:
            gt_label = 1
        elif move_to_place == answer:
            gt_label = 2

        return [gt_label, pred_label]

    def check_answer_for_cg_location(self, prediction: str, answer: str) -> list:
        prediction = prediction.lower()
        answer = answer.lower()

        if "no" in prediction and "yes" not in prediction:
            pred_label = 0
        elif "yes" in prediction and "no" not in prediction:
            pred_label = 1
        else:
            pred_label = -1

        if "no" in answer:
            gt_label = 0
        elif "yes" in answer:
            gt_label = 1

        return [gt_label, pred_label]

    def check_fullness_answer(self, prediction: str, answer: str) -> list:
        prediction = prediction.replace(".", "").lower()
        less_full_answer_list = ["less full", "emptier", "more empty"]
        more_full_answer_list = ["more full", "fuller"]
        pred_label, gt_label = None, None
        for less_full_ans in less_full_answer_list:
            if less_full_ans in prediction:
                pred_label = 1

        if not pred_label:
            for more_full_ans in more_full_answer_list:
                if more_full_ans in prediction:
                    pred_label = 2

        if not pred_label:
            if "equally full" in prediction:
                pred_label = 3

        if not pred_label:
            pred_label = -1  # corrupted

        if answer == "less full":
            gt_label = 1
        elif answer == "more full":
            gt_label = 2
        elif answer == "equally full":
            gt_label = 3

        return [gt_label, pred_label]

    def check_accessibility_answer(self, prediction: str, answer: str) -> list:
        prediction = prediction.replace(".", "").lower()
        pred_label, gt_label = None, None
        if "more accessible" in prediction:
            pred_label = 1
        elif "less accessible" in prediction:
            pred_label = 2
        elif "equally accessible" in prediction:
            pred_label = 3
        else:
            pred_label = -1  # corrupted

        if answer == "more accessible":
            gt_label = 1
        elif answer == "less accessible":
            gt_label = 2
        else:
            gt_label = 3

        return [gt_label, pred_label]

    def check_attitude_answer(self, prediction: str, answer: str) -> list:
        prediction = prediction.lower()
        answer = answer.lower()
        answer_map = {"a": "positive", "b": "neutral", "c": "negative"}
        prediction_token = (
            prediction.split("\n\n")[-1].split(":")[-1].split(".")[0].strip().lower()
        )
        gt_label, pred_label = None, None

        if answer == "positive":
            gt_label = 1
        elif answer == "negative":
            gt_label = 2
        else:
            gt_label = 3

        try:
            prediction = answer_map[prediction_token]
            if prediction == "positive":
                pred_label = 1
            elif prediction == "negative":
                pred_label = 2
            else:
                pred_label = 3

        except:
            if "positive" in prediction_token and "negative" in prediction_token:
                pred_label = -1
            elif "positive" in prediction_token and "neutral" in prediction_token:
                pred_label = -1
            elif "neutral" in prediction_token and "negative" in prediction_token:
                pred_label = -1
            elif "positive" in prediction_token:
                pred_label = 1
            elif "negative" in prediction_token:
                pred_label = 2
            elif "neutral" in prediction_token:
                pred_label = 3
            else:
                pred_label = -1

        return [gt_label, pred_label]
