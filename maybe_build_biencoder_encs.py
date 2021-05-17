"""
Generate examples using generative models
"""
import random
from pathlib import Path

from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript


def setup_args():
    parser = ParlaiParser(
        add_parlai_args=True,
        add_model_args=True,
        description="Maybe generate biencoder embeddings",
    )
    generate = parser.add_argument_group("Generation")
    generate.add_argument("--encs-file-path", type=str)
    return parser


def generate_task(opt):
    opt.log()
    if Path(opt["encs_file_path"]).exists():
        return

    agent = create_agent(opt, requireModelExists=True)
    print("Done.")


class GenerateTask(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return generate_task(self.opt)


if __name__ == '__main__':
    random.seed(42)
    GenerateTask.main()
