import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer, util
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm

from distillation_robustness.paths import MODEL_DIR, TEACHER_OUTPUT_DIR


def evaluate_models():
    """Compare student generations against teacher outputs with embedding similarity."""

    test_data = load_from_disk(str(TEACHER_OUTPUT_DIR / "dolly" / "test"))
    
    student_path = MODEL_DIR / "dolly_filtered_student" / "final"
    s_tokenizer = T5Tokenizer.from_pretrained(str(student_path))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s_model = T5ForConditionalGeneration.from_pretrained(
        str(student_path),
        torch_dtype=torch.bfloat16
    ).to(device)
    s_model.eval()
    
    student_answers = []
    teacher_answers = []

    BATCH_SIZE = 64
    eval_subset = test_data

    for i in tqdm(range(0, len(eval_subset), BATCH_SIZE)):
        batch = eval_subset[i : i + BATCH_SIZE]
        prompts = batch['input']
        
        teacher_answers.extend(batch['teacher_output'])
        
        s_inputs = s_tokenizer(
            prompts, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True,
            padding=True 
        ).to(device)
        
        with torch.no_grad():
            s_out = s_model.generate(
                input_ids=s_inputs.input_ids,
                attention_mask=s_inputs.attention_mask,
                max_length=512,
                num_beams=4,
                early_stopping=True
            )

        batch_responses = s_tokenizer.batch_decode(s_out, skip_special_tokens=True)
        student_answers.extend(batch_responses)

    embedder = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
    
    teacher_embeddings = embedder.encode(teacher_answers, convert_to_tensor=True, batch_size=128)
    student_embeddings = embedder.encode(student_answers, convert_to_tensor=True, batch_size=128)
    
    cosine_scores = util.cos_sim(teacher_embeddings, student_embeddings)
    
    paired_scores = torch.diag(cosine_scores).tolist()
    avg_similarity = sum(paired_scores) / len(paired_scores)
    
    print("\n" + "=" * 40)
    print("EVALUATION RESULTS")
    print("=" * 40)
    print(f"Total pairs evaluated:      {len(paired_scores)}")
    print(f"Average Semantic Similarity: {avg_similarity:.4f} (Max 1.0)")
    print("=" * 40)

if __name__ == "__main__":
    evaluate_models()
