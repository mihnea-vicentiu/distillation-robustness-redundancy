import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from distillation_robustness.paths import PROCESSED_DATA_DIR, TEACHER_OUTPUT_DIR


def generate_teacher_outputs():
    """Generate teacher responses for the Self-Instruct train and validation splits."""

    dataset = load_from_disk(str(PROCESSED_DATA_DIR / "self_instruct_splits"))

    teacher_id = "Qwen/Qwen3.5-4B"

    tokenizer = AutoTokenizer.from_pretrained(teacher_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        teacher_id, 
        device_map="cuda",               
        torch_dtype=torch.bfloat16,      
        attn_implementation="sdpa"
    )
    model.eval()

    BATCH_SIZE = 32

    for split_name in ("train", "validation"):
        split_data = dataset[split_name]
        formatted_prompts = []

        for prompt in split_data['input']:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_prompts.append(text)

        all_responses = []

        for i in tqdm(range(0, len(formatted_prompts), BATCH_SIZE), desc=f"Generating {split_name}"):
            batch_prompts = formatted_prompts[i : i + BATCH_SIZE]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            prompt_length = inputs.input_ids.shape[1]
            generated_ids = outputs[:, prompt_length:]

            responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_responses.extend(responses)

        labeled_dataset = split_data.add_column("teacher_output", all_responses)
        labeled_dataset.save_to_disk(str(TEACHER_OUTPUT_DIR / "self_instruct" / split_name))

if __name__ == "__main__":
    torch.cuda.empty_cache()
    generate_teacher_outputs()
