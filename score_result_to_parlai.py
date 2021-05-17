"""
Port score result into parlai dialogue data format
"""
import argparse
import json
from collections import defaultdict


def _examples_to_parlai_dialogue(examples, only_original, score_name, normalize_scores=False):
    assert len(examples) > 0

    parlai_dialogue = ""

    context = examples[0]["context"]
    text_str = "text:" + "\\n".join(context)
    parlai_dialogue += text_str

    label = None
    if only_original:
        for example in examples:
            if example["source"] == "original":
                label = example["response"]
                break
        if label is None:
            # No original
            return None, None
        responses = [label]
    else:
        # label with largest score
        if score_name is not None:
            label = max(examples, key=lambda x: x["score_infos"][score_name])["response"]
        else:
            label = examples[0]["response"]
        responses = [example["response"] for example in examples]
    parlai_dialogue += f"\tlabels:{label}"

    # Add candidates
    if not only_original:
        candidates_str = "\tlabel_candidates:" + "|".join(
            example["response"].replace("|", "") for example in examples)
        parlai_dialogue += candidates_str

        if score_name is not None:
            scores = [example["score_infos"][score_name] for example in examples]
            if normalize_scores:
                # Normalize scores between 0 and 1
                score_max = max(scores)
                score_min = min(scores)
                scores = [(score - score_min) / (score_max - score_min) for score in scores]
        else:
            scores = [1.0 for example in examples]

        candidates_scores_str = "\tlabel_candidates_scores:" + "|".join(
            f"{score:.8f}" for score in scores
        )
        parlai_dialogue += candidates_scores_str

        candidates_masks_str = "\tlabel_candidates_masks:" + "|".join(
            "1" for example in examples
        )
        parlai_dialogue += candidates_masks_str

    parlai_dialogue += "\tepisode_done:True"
    return parlai_dialogue, responses


def normalize_response(response):
    response = response.replace("|", "")
    response = response.replace("\n", " ")
    response = response.replace("\t", " ")
    return response


def main(args):
    with open(args.input_path) as f:
        examples = [json.loads(line.strip()) for line in f]

    # merge examples with same context
    context_to_examples = defaultdict(list)
    for example in examples:
        example["response"] = normalize_response(example["response"])
        context_to_examples[tuple(example["context"])].append(example)

    output_lines = []

    for _examples in context_to_examples.values():
        output_line, _responses = _examples_to_parlai_dialogue(
            _examples, args.only_original, args.score_name)
        _output_lines = [output_line]
        if output_lines is not None:
            output_lines.extend(_output_lines)

    with open(args.output_parlai_path, "w") as f:
        for output_line in output_lines:
            f.write(output_line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-parlai-path", type=str,
                        help="Path for saving into ParlAI format")
    parser.add_argument("--only-original", action="store_true")
    parser.add_argument("--score-name", type=str, default=None)

    args = parser.parse_args()
    main(args)
