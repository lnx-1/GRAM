import argparse
import json
import os

import torch
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-m", "--model", required=True)
parser.add_argument("-o", "--output", required=True)
parser.add_argument("-b", "--batch-size", type=int, default=1)
parser.add_argument("-bm", "--base-model", default=None)
parser.add_argument("--latent-dim", type=int, default=64)
parser.add_argument("--mc-samples", type=int, default=8)
parser.add_argument("--logvar-min", type=float, default=-8.0)
parser.add_argument("--logvar-max", type=float, default=4.0)
parser.add_argument("--label-a", default="A")
parser.add_argument("--label-b", default="B")
args = parser.parse_args()

if os.path.exists(args.output):
    os.remove(args.output)

system_prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better.
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible.
Please directly output your final verdict by strictly following this format: "A" if assistant A is better, "B" if assistant B is better.

[User Question]
{input}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]

#Preferred: """

tokenizer_path = args.base_model or args.model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer.padding_side = "left"
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()

label_token_ids = []
for label in [args.label_a, args.label_b]:
    token_ids = tokenizer(label, add_special_tokens=False).input_ids
    if len(token_ids) != 1:
        raise ValueError(f"Label must map to one token, got {label}: {token_ids}")
    label_token_ids.append(token_ids[0])

label_token_ids = torch.tensor(label_token_ids, device=model.device)
target_order = torch.tensor([[0, 1], [1, 0]], device=model.device)


def apply_chat_template(message):
    try:
        return tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)


def project_label_logits(hidden_states):
    output_embeddings = model.get_output_embeddings()
    label_weights = output_embeddings.weight.index_select(
        dim=0, index=label_token_ids.to(output_embeddings.weight.device)
    ).to(hidden_states.device)
    label_weights = label_weights.to(hidden_states.dtype)
    logits = torch.matmul(hidden_states, label_weights.transpose(0, 1))
    bias = getattr(output_embeddings, "bias", None)
    if bias is not None:
        label_bias = bias.index_select(dim=0, index=label_token_ids.to(bias.device)).to(hidden_states.device)
        logits = logits + label_bias.to(logits.dtype)

    return logits


def compute_vgrm_scores(outputs, batch_size):
    decision_states = outputs.hidden_states[-1][:, -1, :]
    hidden_size = decision_states.size(-1)
    if args.latent_dim * 2 > hidden_size:
        raise ValueError(f"`latent_dim * 2` must be <= hidden size, got {args.latent_dim} * 2 > {hidden_size}.")

    mu = decision_states[:, : args.latent_dim].float()
    logvar = decision_states[:, args.latent_dim: 2 * args.latent_dim].float().clamp(args.logvar_min, args.logvar_max)
    std = torch.exp(0.5 * logvar)
    mc_samples = max(1, args.mc_samples)
    eps = torch.randn(decision_states.size(0), mc_samples, args.latent_dim, device=decision_states.device)
    sampled_z = mu.unsqueeze(1) + std.unsqueeze(1) * eps
    sampled_states = decision_states.float().unsqueeze(1).expand(-1, mc_samples, -1).clone()
    sampled_states[:, :, : args.latent_dim] = sampled_z

    logits = project_label_logits(sampled_states.to(decision_states.dtype)).float()
    row_order = target_order.repeat(batch_size, 1)
    ordered_logits = torch.gather(logits, 2, row_order.unsqueeze(1).expand(-1, mc_samples, -1))
    probs = torch.softmax(ordered_logits, dim=-1)
    mean_probs = probs.mean(dim=1).view(batch_size, 2, 2)
    scores = mean_probs.mean(dim=1)

    mc_variance = probs.var(dim=1, unbiased=False).view(batch_size, 2, 2).mean(dim=1)
    latent_uncertainty = logvar.exp().mean(dim=-1).view(batch_size, 2).mean(dim=1)
    entropy = -(scores.clamp_min(1e-8) * scores.clamp_min(1e-8).log()).sum(dim=-1)
    uncertainty = entropy + mc_variance.mean(dim=-1) + latent_uncertainty
    return scores, mc_variance, latent_uncertainty, entropy, uncertainty


with open(args.input, "r", encoding="utf-8") as f:
    input_data = json.load(f)

for idx in trange(0, len(input_data), args.batch_size):
    res = []
    batch_data = input_data[idx: idx + args.batch_size]
    current_batch_size = len(batch_data)
    messages = []
    for item in batch_data:
        messages += [
            [{"role": "user", "content": system_prompt.format(input=item["prompt"], response_a=item["chosen"], response_b=item["rejected"])}],
            [{"role": "user", "content": system_prompt.format(input=item["prompt"], response_a=item["rejected"], response_b=item["chosen"])}],
        ]

    prompt = [apply_chat_template(message) for message in messages]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
        scores, mc_variance, latent_uncertainty, entropy, uncertainty = compute_vgrm_scores(
            output, current_batch_size
        )

    for row_idx, data_item in enumerate(batch_data):
        data_item["score_chosen"] = float(scores[row_idx, 0].item())
        data_item["score_rejected"] = float(scores[row_idx, 1].item())
        data_item["correct"] = data_item["score_chosen"] > data_item["score_rejected"]
        data_item["mc_variance_chosen"] = float(mc_variance[row_idx, 0].item())
        data_item["mc_variance_rejected"] = float(mc_variance[row_idx, 1].item())
        data_item["latent_uncertainty"] = float(latent_uncertainty[row_idx].item())
        data_item["predictive_entropy"] = float(entropy[row_idx].item())
        data_item["uncertainty"] = float(uncertainty[row_idx].item())
        res.append(data_item)

    mode = "a" if os.path.exists(args.output) else "w"
    with open(args.output, mode, encoding="utf-8") as f:
        for item in res:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
