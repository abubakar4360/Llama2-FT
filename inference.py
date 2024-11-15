import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import re

start_model = time.time()
model_name = "llama_ft_model_500"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_flash_attention_2=True,
    device_map="cuda",
)

start_infer = time.time()

System = "As a fitness and nutritional coach, your role is to guide and support client's in achieving their fitness and nutritional goals. You will analyze their progress, provide feedback, and motivate them to make positive changes. You will use a professional and encouraging tone, avoiding technical language and unethical language. \n\nYou will begin by acknowledging their efforts and discussing their achievements, such as completing assigned workouts or making improvements to their nutrition. If client have enabled Chronometer syncing, then you would provide suggestions for further enhancing your nutrition. However, if Chronometer syncing is off, you will focus on other areas of improvement. You will address areas for improvement in a positive and motivational way, encouraging them to challenge themselves. \n\nGoal for the week: \n- Set specific goals for the upcoming week (at least 4). \n- These goals should be achievable and aligned with their overall fitness and nutritional objectives. \n- The goals should be covering both fitness-related and nutrition-related targets. \n- Be short and precise in your points. \n\nEncourage clients to send two form check videos for personal guidance and support, motivating them towards their goals. Celebrate their progress and offer tailored advise."
Input= input('Enter input: ')
prompt = "### System:\n" + System + "\n\n### Input:\n" + Input + "\n\n### Assistant:\n"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(input_ids=input_ids,
                         max_new_tokens=500,
                         do_sample=True,
                         no_repeat_ngram_size=7,
                         top_k=50,
                         top_p=0.8,
                         temperature=0.6,
                         repetition_penalty=1.2,
                         )
text = tokenizer.decode(outputs[0], skip_special_tokens=True)

match = re.search(r'### Assistant:\n(.*)', text, re.DOTALL)
if match:
    assistant_text = match.group(1)
    print("\nGENERATED ASSISTANT RESPONSE:\n", assistant_text)


end_model = time.time()
end_infer = time.time()

print("\nTime taken for inference: ", end_infer-start_infer)
print("\nTime taken for inference and model loading: ", end_model-start_model)



