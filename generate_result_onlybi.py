"""
Generate examples using generative models
"""
import json
import math
import random
import time

import faiss
import numpy as np
import torch
from parlai.core.agents import create_agent, create_agent_from_model_file
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.strings import normalize_reply


def setup_args():
    parser = ParlaiParser(
        add_parlai_args=True,
        add_model_args=True,
        description="Generate with our full pipeline. ",
    )

    generate = parser.add_argument_group("Generation")
    generate.add_argument("--input-path", type=str)
    generate.add_argument("--faiss-index-path", type=str)
    generate.add_argument("--faiss-efsearch", type=int, default=256)
    generate.add_argument("--biencoder-model-file", type=str)
    generate.add_argument("--search-topk", type=int, default=128)
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


def encode_context(agent, context):
    _update_history(agent, context)
    with torch.no_grad():
        batch = agent.batchify([agent.observation])
        context_h, _ = agent.model(xs=batch.text_vec, mems=None, cands=None)
        context_h = context_h.float().cpu().numpy()
    return context_h


def generate_result(agent, context, topk_indices):
    _update_history(agent, context)
    with torch.no_grad():
        batch = agent.batchify([agent.observation])
        ctxt_rep, ctxt_rep_mask, _ = agent.model(ctxt_tokens=batch.text_vec)
        topk_indices = torch.LongTensor(topk_indices).to("cuda")
        cands_h = agent.fixed_candidate_encs[0][topk_indices].unsqueeze(0)
        scores = agent.model(
            ctxt_rep=ctxt_rep, ctxt_rep_mask=ctxt_rep_mask, cand_rep=cands_h,
        )[0]
        argmax_idx = topk_indices[scores.argmax().item()]

    return agent.fixed_candidates[argmax_idx]


def _get_index_key_based_on_embedding_num(num_embeddings: int) -> str:
    """https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index"""
    if num_embeddings < 1e6:
        k = 8 * math.sqrt(num_embeddings)
        k_power_2 = 2 ** math.floor(math.log2(k))
        return f"IVF{k_power_2},Flat"
    elif num_embeddings < 1e7:
        return "IVF65536_HNSW32,Flat"
    elif num_embeddings < 1e8:
        return "IVF262144_HNSW32,Flat"
    else:
        return "IVF1048576_HNSW32,Flat"


def build_faiss_index(embedding_encs):
    num_candidates, embedding_dim = embedding_encs.shape
    index = faiss.index_factory(
        embedding_dim,
        "HNSW32,Flat",
        faiss.METRIC_INNER_PRODUCT,
    )
    embedding_encs = embedding_encs.cpu().numpy().astype(np.float32)
    index.add(embedding_encs)
    return index


def generate_response(biencoder_agent, faiss_index, context, topk):
    context_h = encode_context(biencoder_agent, context)
    scores, topk_indices = faiss_index.search(context_h, topk)
    scores = scores[0].tolist()
    topk_indices = topk_indices[0].tolist()

    max_score, max_index = max(list(zip(scores, topk_indices)), key=lambda x: x[0])
    return biencoder_agent.fixed_candidates[max_index]


def generate_task(opt):
    opt.log()
    print("Load Biencoder Agent")
    biencoder_agent = create_agent_from_model_file(
        opt["biencoder_model_file"],
        opt_overrides={
            "eval_candidates": "fixed",
            "encode_candidate_vecs_batchsize": opt["encode_candidate_vecs_batchsize"],
            "fixed_candidates_path": opt["fixed_candidates_path"],
        }
    )
    print("Load Faiss Index")
    faiss_index = faiss.read_index(opt["faiss_index_path"])
    faiss_index.hnsw.efSearch = opt["faiss_efsearch"]

    with open(opt["input_path"]) as f:
        dialogues = [json.loads(line.strip()) for line in f]

    # Warmup
    warmup_num = 3
    for _ in range(warmup_num):
        _ = generate_response(
            biencoder_agent, faiss_index,
            dialogues[0], opt["search_topk"],
        )

    times = []
    results = []

    for dialogue in dialogues:
        start_time = time.time()
        result_text = generate_response(
            biencoder_agent, faiss_index,
            dialogue, opt["search_topk"],
        )
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
