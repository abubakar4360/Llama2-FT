from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "meta-llama/Llama-2-7b-chat-hf"
weights = "llama_ft_weights_500/checkpoint-155"
new_model_name = "llama_ft_model_500"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
base_model_reload = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    device_map="cuda",
    use_flash_attention_2=True,
)

model = PeftModel.from_pretrained(base_model_reload, weights)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained(new_model_name,safe_serialization=True)
tokenizer.save_pretrained(new_model_name)
