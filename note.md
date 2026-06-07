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