import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from distillation_robustness.paths import SCORED_DATA_DIR, TEACHER_OUTPUT_DIR


def compute_sequence_log_prob(model, input_ids, prompt_length):
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, :-1, :]
        labels = input_ids[0, 1:]
        
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        
        response_log_probs = token_log_probs[prompt_length - 1:]
        
        return response_log_probs.mean().item()

def score_dataset():
    """Score teacher outputs against a smaller reference model."""

    dataset = load_from_disk(str(TEACHER_OUTPUT_DIR / "self_instruct" / "train"))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(teacher_id)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_id, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()
    
    print("Loading Qwen3.5-0.8B (Reference)...")
    ref_id = "Qwen/Qwen3.5-0.8B"
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_id, device_map=device, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).eval()

    teacher_scores = []
    ref_scores = []

    for sample in tqdm(dataset, desc="Scoring Samples"):
        prompt = sample['input']
        response = sample['teacher_output']
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_text = formatted_prompt + response
        
        prompt_length = len(tokenizer(formatted_prompt).input_ids)
        full_input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(device)

        t_score = compute_sequence_log_prob(teacher_model, full_input_ids, prompt_length)
        r_score = compute_sequence_log_prob(ref_model, full_input_ids, prompt_length)
        
        teacher_scores.append(t_score)
        ref_scores.append(r_score)

    dataset = dataset.add_column("teacher_score", teacher_scores)
    dataset = dataset.add_column("ref_score", ref_scores)
    
    dataset.save_to_disk(str(SCORED_DATA_DIR / "self_instruct_teacher_scores"))
    print("Success! Dataset scored and saved.")

if __name__ == "__main__":
    score_dataset()
