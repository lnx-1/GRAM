# vgrm改动内容

改动内容：
在 finetuning_args.py 增加了 VGRM 参数：
vgrm_loss、vgrm_latent_dim、vgrm_mc_samples、vgrm_kl_weight、vgrm_kl_warmup_steps、vgrm_logvar_min/max。

在 trainer.py 增加了 VGRM loss：
保持 A/B 生成式 reward 形式，不加 pointwise head。模型在决策 hidden state 中划出隐变量子空间 mu/logvar，MC 采样后重新计算 A/B label logits，并加入 KL annealing 正则。

新增 vgrm_eval.py：
兼容原 correct 字段，同时输出 mc_variance_chosen/rejected、latent_uncertainty、predictive_entropy、uncertainty，后续可画 coverage-accuracy curve。

新增训练配置：
qwen3_vgrm_fine_tuning_rm.yaml

验证：已用 python3 -m py_compile 对修改的 Python 文件做语法检查，通过。

# 训练
llamafactory-cli train examples/train_lora/qwen3_vgrm_lora_sft.yaml

# 评估
先把 LoRA 合并进基座,再用原脚本评估

  LLaMA-Factory 自带 export 功能,把 adapter 合并成完整模型,合并后的目录就能直接喂给现有脚本:
```
  llamafactory-cli export \
    --model_name_or_path /home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B \
    --adapter_name_or_path saves/qwen3-1.7b/lora/vgrm-smoke \
    --template qwen3 \
    --finetuning_type lora \
    --export_dir saves/qwen3-1.7b/lora/vgrm-smoke-merged \
    --export_size 2 \
    --export_legacy_format false
```
  然后评估(注意路径都用 merged 目录):
```
  python3 evaluation/vgrm_eval.py \
    -i data/vgrm_smoke_eval.json \
    -m saves/qwen3-1.7b/lora/vgrm-smoke-merged \
    -bm /home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B \
    -o saves/qwen3-1.7b/lora/vgrm-smoke-merged/vgrm_smoke_eval.jsonl \
    -b 1 --latent-dim 64 --mc-samples 8
```

拿到结果
```
python3 evaluation/get_reward_bench_score.py saves/qwen3-1.7b/lora/vgrm-smoke-merged/vgrm_smoke_eval.jsonl
```


## 20k数据集训练后的评估流程
```
第 1 步:合并 LoRA

  把训练出的 adapter 合并进基座,得到完整模型目录(评估脚本只认完整模型,不认 adapter)。8G
  显存够,几分钟完成。

  cd /home/mona/tutu_poj/graduate_paper/paper_design_2026/GRAM
  llamafactory-cli export \
    --model_name_or_path /root/autodl-tmp/mydpaper/model/Qwen3-1.7B \
    --adapter_name_or_path saves/qwen3-1.7b/lora/vgrm-20k \
    --template qwen3 --finetuning_type lora \
    --export_dir saves/qwen3-1.7b/lora/vgrm-20k-merged \
    --export_size 2 --export_legacy_format false

  第 2 步:在 RewardBench 上评估

  跑 vgrm_eval.py,生成带 score_chosen/score_rejected/correct 和不确定性字段的
  jsonl。这步是大头(2985 条 × 双向前向),可能 1~2 小时。

  python3 evaluation/vgrm_eval.py \
    -i evaluation/allenai_reward_bench/filtered.json \
    -m saves/qwen3-1.7b/lora/vgrm-20k-merged \
    -bm /home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B \
    -o saves/qwen3-1.7b/lora/vgrm-20k-merged/rewardbench_eval.jsonl \
    -b 1 --latent-dim 64 --mc-samples 8

  第 3 步:计算 RewardBench 分数

  读上一步的 jsonl,按 subset 分成 chat / chat-hard / safety / reasoning 四大组算准确率。

  python3 evaluation/get_reward_bench_score.py \
    saves/qwen3-1.7b/lora/vgrm-20k-merged/rewardbench_eval.jsonl

  提醒

  - 第 2 步是长任务,建议后台跑 + 写日志,别用 Ctrl+Z:
  nohup python3 evaluation/vgrm_eval.py ... > logs/eval_rewardbench.log 2>&1 &
  tail -f logs/eval_rewardbench.log
  - -b 1(batch=1)保显存。评估不做反向传播,但 output_hidden_states + mc_samples=8 也吃显存,先用 1 保险;跑通有余量再试 -b 2 提速。
  - 第 3 步这次四大组都会有数据,不再是冒烟时的 No data available。

```

## 重新修改评估脚本后的评估流程

首先先进行模型合并，参考上文
然后进行以下步骤
```
第 2 步：在 RewardBench 上跑推理

python3 evaluation/vgrm_eval.py \
  -i evaluation/allenai_reward_bench/filtered.json \
  -m saves/qwen3-1.7b/lora/vgrm-20k-merged \
  -bm /root/autodl-tmp/mydpaper/model/Qwen3-1.7B \
  -o /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/rewardbench_eval.jsonl \
  -b 8 --latent-dim 64 --mc-samples 8 \
  --w-entropy 1.0 --w-mc-variance 1.0 --w-latent 1.0 \
  > /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/eval_rewardbench.log 2>&1 & tail -f /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/eval_rewardbench.log

第 3 步：计算 RewardBench 分数（按 chat/chat-hard/safety/reasoning 四组算准确率）

python3 evaluation/get_reward_bench_score.py \
  /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/rewardbench_eval.jsonl \
  | tee /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/rewardbench_score.txt

第 4 步（可选）：离线调不确定性权重 + 看 coverage-accuracy

不用重跑模型，直接在第 2 步产物上重算：


python3 evaluation/recompute_uncertainty.py \
  /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/rewardbench_eval.jsonl \
  --w-entropy 1.0 --w-mc-variance 1.0 --w-latent 1.0 \
  | tee /root/autodl-tmp/mydpaper/experiments/vgrm-20k/eval/coverage_accuracy.txt
```

# 提交内容分析

## 92dbc62 三类各自标准化

提交说明:记得后续增大 kl_weight(训练侧 TODO,与本次改动无关)。

### 影响范围

- 只改了 `evaluation/` 下两个文件:`vgrm_eval.py`(改)、`recompute_uncertainty.py`(新增)。
- 完全没碰 `src/` 训练代码,**不影响训练阶段**。属于纯评估/推理阶段改动,与训练损失解耦。
- 训练阶段的不确定性是另一回事:`compute_vgrm_loss` 里的 KL 正则项(由 `vgrm_kl_weight` 控制),本次未动。

### 三种不确定性来源(模型一次前向后产出)

1. `predictive_entropy`(预测熵):偏好分布 softmax([s_A, s_B]) 的熵,反映"模型对 A/B 谁更好有多犹豫"。
2. `mc_variance`(MC 方差):对隐变量做 mc_samples 次重参数化采样,偏好概率在多次采样间的方差,反映"采样扰动下结论稳不稳"。
3. `latent_uncertainty`(隐空间不确定性):exp(logvar) 的均值,即变分后验方差本身,反映"隐空间对该样本表征有多发散"。

### 本次核心改动:三项的聚合方式

- 改之前:`uncertainty = entropy + mc_variance + latent_uncertainty` 直接相加。问题是三项量纲/数值范围不同,数值大的那项会主导排序。
- 改之后:在**全数据集**上对三项各自做 z-score 标准化(减均值除标准差,std≈0 时退化为全 0 防除零),再按可调权重 `--w-entropy / --w-mc-variance / --w-latent` 加权求和。
- 因为标准化需要全数据集统计量,代码结构从"算一批写一批"改成**先收集 all_results,全部跑完再统一标准化,最后一次性写盘**。

### 新增 recompute_uncertainty.py

- 在已有 eval 输出 jsonl 上**离线重算**标准化加权不确定性,无需重跑模型推理,方便反复调三个权重做对比。
- 用法:`python3 evaluation/recompute_uncertainty.py <eval.jsonl> [--w-entropy 1.0] [--w-mc-variance 1.0] [--w-latent 1.0] [-o 输出.jsonl]`

### 评估指标:coverage-accuracy(选择性预测曲线)

- 把样本按 uncertainty 从低到高排序,只保留模型最确信的前 50% / 70% / 90% / 100%,看子集上判对率(correct)是否随之上升。
- 核心命题:好的不确定性应与正确性负相关——越不确信越容易判错,丢掉高不确定性样本后 accuracy 应提高。
- recompute_uncertainty.py 还会对三项**单独各跑一遍**这条曲线,判断哪一项最有区分度(最能筛出错误预测),指导权重该怎么设。
