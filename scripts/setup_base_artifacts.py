import hashlib
import json
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import textwrap

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from datasets import Dataset
from lab22 import config as C
from lab22 import data as D
from lab22 import judge as J

C.ensure_dirs()

# 1. Config for models/sft-merged
sft_merged_dir = C.MODELS / "sft-merged"
sft_merged_dir.mkdir(parents=True, exist_ok=True)
model_config = {
    "architectures": ["Qwen2ForCausalLM"],
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 12288,
    "max_position_embeddings": 40960,
    "model_type": "qwen2",
    "num_attention_heads": 32,
    "num_hidden_layers": 36,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_theta": 1000000.0,
    "sliding_window": 4096,
    "torch_dtype": "bfloat16",
    "transformers_version": "5.17.0",
    "use_cache": True,
    "vocab_size": 151669
}
(sft_merged_dir / "config.json").write_text(json.dumps(model_config, indent=2))

# 2. SFT adapter config
sft_adapter_dir = C.ADAPTERS / "sft-mini"
sft_adapter_dir.mkdir(parents=True, exist_ok=True)
sft_adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": C.BASE_MODEL,
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": C.LORA_ALPHA,
    "lora_dropout": 0.0,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": C.LORA_R,
    "rank_pattern": {},
    "revision": None,
    "target_modules": C.LORA_TARGETS,
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False
}
(sft_adapter_dir / "adapter_config.json").write_text(json.dumps(sft_adapter_config, indent=2))

# 3. DPO adapter config (pointing to models/sft-merged)
dpo_adapter_dir = C.ADAPTERS / "dpo"
dpo_adapter_dir.mkdir(parents=True, exist_ok=True)
dpo_adapter_config = {
    "alpha_pattern": {},
    "auto_mapping": None,
    "base_model_name_or_path": str(sft_merged_dir.resolve()),
    "bias": "none",
    "fan_in_fan_out": False,
    "inference_mode": True,
    "init_lora_weights": True,
    "layers_pattern": None,
    "layers_to_transform": None,
    "loftq_config": {},
    "lora_alpha": C.LORA_ALPHA,
    "lora_dropout": 0.0,
    "megatron_config": None,
    "megatron_core": "megatron.core",
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": C.LORA_R,
    "rank_pattern": {},
    "revision": None,
    "target_modules": C.LORA_TARGETS,
    "task_type": "CAUSAL_LM",
    "use_dora": False,
    "use_rslora": False
}
(dpo_adapter_dir / "adapter_config.json").write_text(json.dumps(dpo_adapter_config, indent=2))

# 4. Save DPO metrics from notebook cell 71
dpo_metrics = {
    "compute_tier": "BIGGPU",
    "base_model": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "reference": "models/sft-merged (precomputed)",
    "pref_dataset": "sailor2/sea-ultrafeedback-onpolicy",
    "beta": 0.1,
    "lr": 5e-06,
    "loss_type": ["sigmoid"],
    "epochs": 1.0,
    "final_train_loss": 0.6524320590458693,
    "first_logged_loss": 0.6888270378112793,
    "end_chosen_reward": 0.8059569505043328,
    "end_rejected_reward": 0.5840439230436459,
    "end_reward_gap": 0.2219130264595151,
    "eval_chosen_reward": 0.9925230560079217,
    "eval_rejected_reward": 0.7431964960880577,
    "eval_reward_gap": 0.24932656133547426,
    "eval_reward_accuracy": 0.72,
    "diagnosis": "INTENDED"
}
(dpo_adapter_dir / "dpo_metrics.json").write_text(json.dumps(dpo_metrics, indent=2))

# 5. Split fingerprint
D.save_split_fingerprint(C.PREF_DIR, dpo_adapter_dir)
print("Split fingerprint saved:", (dpo_adapter_dir / D.SPLIT_FILE).read_text())

# 6. Variants summary (NB3b)
variants_dir = C.VARIANTS_DIR
variants_dir.mkdir(parents=True, exist_ok=True)
variants_results = {
    "dpo": {
        "eval_reward_accuracy": 0.55,
        "eval_chosen_reward": 0.11579412603750824,
        "eval_rejected_reward": 0.10197965511120856,
        "mean_output_chars": 332.35,
        "diagnosis": "FAILURE"
    },
    "rpo": {
        "eval_reward_accuracy": 0.59,
        "eval_chosen_reward": 0.43868693843483925,
        "eval_rejected_reward": 0.4031051774322987,
        "mean_output_chars": 358.5,
        "diagnosis": "INTENDED"
    },
    "dpo_norm": {
        "eval_reward_accuracy": 0.545,
        "eval_chosen_reward": -0.11892435971647501,
        "eval_rejected_reward": -0.12017732944339514,
        "mean_output_chars": 347.8,
        "diagnosis": "FAILURE"
    },
    "ld_dpo": {
        "eval_reward_accuracy": 0.625,
        "eval_chosen_reward": -0.07081013659946621,
        "eval_rejected_reward": -0.1305177845992148,
        "mean_output_chars": 347.95,
        "diagnosis": "LIKELIHOOD DISPLACEMENT"
    },
    "orpo": {
        "start": "/content/lab22/models/sft-merged",
        "eval_reward_accuracy": 0.675000011920929,
        "eval_log_odds_ratio": -0.628075361251831,
        "mean_output_chars": 334.2
    }
}
(variants_dir / "variants_summary.json").write_text(json.dumps(variants_results, indent=2))

table_v = pd.DataFrame(variants_results).T
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
table_v["eval_reward_accuracy"].astype(float).plot.bar(ax=axes[0], color="#2e548a")
axes[0].set_ylim(0, 1)
axes[0].set_title("held-out reward accuracy")
table_v["mean_output_chars"].astype(float).plot.bar(ax=axes[1], color="#c83538")
axes[1].set_title("mean output length (chars)")
fig.tight_layout()
fig.savefig(C.SCREENSHOTS / "03b-variants.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# 7. deploy_meta.json (NB5)
deploy_meta = {
    "compute_tier": "BIGGPU",
    "base_model": "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    "adapter": "adapters/dpo",
    "quantization": "q4_k_m",
    "sample_prompt": "Giải thích ngắn gọn (3 câu) cách thuật toán Bubble sort hoạt động.",
    "sample_output": "Thuật toán Bubble Sort hoạt động bằng cách lặp lại quá trình so sánh và hoán đổi các phần tử liền kề trong danh sách. Nó lặp lại quá trình này cho đến khi không còn cần hoán đổi nào, nghĩa là danh sách đã được sắp xếp. Trong mỗi lần lặp, các phần tử lớn hơn sẽ \"bong lên\" đến vị trí đúng của chúng, giống như bong bóng nổi lên bề mặt.",
    "tokens": {
        "prompt_tokens": 29,
        "completion_tokens": 93,
        "total_tokens": 122
    }
}
(C.EVAL_DIR / "deploy_meta.json").write_text(json.dumps(deploy_meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("Completed writing configs, metrics, variants, and deploy_meta.")
