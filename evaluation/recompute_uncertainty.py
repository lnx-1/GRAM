"""在已有的 vgrm_eval 输出 jsonl 上，对三项不确定性做 z-score 标准化后加权求和，
重算 uncertainty，并打印 coverage-accuracy 曲线。无需重跑模型推理。

用法:
    python3 evaluation/recompute_uncertainty.py <eval.jsonl> \
        [--w-entropy 1.0] [--w-mc-variance 1.0] [--w-latent 1.0] [-o 输出.jsonl]
"""
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("-o", "--output", default=None, help="写回标准化结果；不指定则只打印分析")
parser.add_argument("--w-entropy", type=float, default=1.0)
parser.add_argument("--w-mc-variance", type=float, default=1.0)
parser.add_argument("--w-latent", type=float, default=1.0)
args = parser.parse_args()

rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def mc_scalar(r):
    if "mc_variance" in r:
        return r["mc_variance"]
    return 0.5 * (r.get("mc_variance_chosen", 0.0) + r.get("mc_variance_rejected", 0.0))


z_ent = standardize([r["predictive_entropy"] for r in rows])
z_mc = standardize([mc_scalar(r) for r in rows])
z_lat = standardize([r["latent_uncertainty"] for r in rows])

for r, ze, zm, zl in zip(rows, z_ent, z_mc, z_lat):
    r["z_predictive_entropy"] = ze
    r["z_mc_variance"] = zm
    r["z_latent_uncertainty"] = zl
    r["uncertainty"] = args.w_entropy * ze + args.w_mc_variance * zm + args.w_latent * zl


def coverage_accuracy(key, label):
    ordered = sorted(rows, key=lambda r: r[key])
    print(f"  [{label}]")
    for cov in [0.5, 0.7, 0.9, 1.0]:
        n = int(len(ordered) * cov)
        acc = sum(r["correct"] for r in ordered[:n]) / n
        print(f"    保留最低 {int(cov*100):>3}%: acc={acc:.4f} (n={n})")


print(f"总样本: {len(rows)}  权重: entropy={args.w_entropy} mc={args.w_mc_variance} latent={args.w_latent}")
print("=== Coverage-Accuracy（标准化加权后的 uncertainty）===")
coverage_accuracy("uncertainty", "weighted z-score")
print("=== 单项各自的 Coverage-Accuracy（用于对比哪一项最有区分度）===")
for key, name in [("z_predictive_entropy", "仅 entropy"),
                  ("z_mc_variance", "仅 mc_variance"),
                  ("z_latent_uncertainty", "仅 latent")]:
    coverage_accuracy(key, name)

if args.output:
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n已写回: {args.output}")
