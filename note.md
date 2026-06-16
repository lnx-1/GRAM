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
    --model_name_or_path /home/mona/tutu_poj/graduate_paper/gram_modeldata/Qwen3-1.7B \
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