"""
Generate examples using basic parlai models
"""
import json
import random
import time

import numpy as np
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import normalize_reply


def setup_args():
    parser = ParlaiParser(
        add_parlai_args=True,
        add_model_args=True,
        description="Generate using parlai agent",
    )

    generate = parser.add_argument_group("Generation")
    generate.add_argument("--input-path")
    generate.add_argument("--result-save-path", type=str)
    return parser


def _update_history(agent, context):
    agent.reset()
    for idx, utterance in enumerate(context):
        from_user = (len(context) - idx) % 2 == 1
        observe_input = {"text": utterance, "episode_done": False}

        # consider as bot
        if not from_user:
            # Initial utterance case
            if agent.observation is None:
                agent.history.add_reply(utterance)
            else:
                agent.self_observe(observe_input)
        else:
            agent.observe(observe_input)


def generate_result(agent, context):
    _update_history(agent, context)
    action = agent.act()

    if "beam_texts" in action:
       return max(action["beam_texts"], key=lambda x: float(x[1]))[0]
    else:
        return action["text"]


def generate_task(opt):
    opt.log()
    agent = create_agent(opt, requireModelExists=True)
    agent.skip_generation = False

    with open(opt["input_path"]) as f:
        dialogues = [json.loads(line.strip()) for line in f]

    # Warmup
    warmup_num = 3
    for _ in range(warmup_num):
        _ = generate_result(agent, dialogues[0])

    times = []
    results = []

    for dialogue in dialogues:
        start_time = time.time()
        result_text = generate_result(agent, dialogue)
        end_time = time.time()

        times.append(end_time - start_time)
        results.append({
            "context": dialogue,
            "response": normalize_reply(result_text),
        })

    avg_time = np.array(times).mean()
    print("Done.")
    print(f"Consumed time avg: {avg_time:.4f} sec")

    with open(opt["result_save_path"], "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


class GenerateTask(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return generate_task(self.opt)


if __name__ == '__main__':
    random.seed(42)
    GenerateTask.main()
