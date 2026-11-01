"""
Create LoRA adapters with exactly the training configuration of train_pred_llm.py
(GemmaConfig: r=8, alpha=8, dropout=0, language attn+mlp layers, vision frozen),
saved the same way (model.save_pretrained). LoRA B is zero-initialised, so the
adapter costs the same compute as the real fine-tuned one but outputs equal the base model.
"""
import os
import sys
import unsloth  # noqa: F401  (must be imported before transformers)
from unsloth import FastVisionModel

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adapters")

for base in sys.argv[1:]:
    save_name = os.path.join(OUT, base.split("/")[-1] + "_task1_zeroLoRA")
    if os.path.exists(os.path.join(save_name, "adapter_config.json")):
        print("exists:", save_name)
        continue
    model, tokenizer = FastVisionModel.from_pretrained(
        base, load_in_4bit=True, use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8, lora_alpha=8, lora_dropout=0, bias="none",
        random_state=99, use_rslora=False, loftq_config=None,
    )
    model.save_pretrained(save_name)
    tokenizer.save_pretrained(save_name)
    print("saved:", save_name)
    with open(os.path.join(save_name, "adapter_config.json")) as f:
        print(f.read())
    del model, tokenizer
