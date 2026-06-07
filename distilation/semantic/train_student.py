import torch
from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments
)

def train_student():
    train_data = load_from_disk("./semantic/self_instruct_teacher/self_instruct_train")
    val_data = load_from_disk("./semantic/self_instruct_teacher/self_instruct_validation")
    
    student_id = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(student_id)
    model = T5ForConditionalGeneration.from_pretrained(student_id)

    def tokenize_function(examples, use_teacher=False):
        inputs = tokenizer(
            examples["input"], 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
        
        target_col = "teacher_output" if use_teacher else "output"
        targets = tokenizer(
            examples[target_col], 
            max_length=512, 
            truncation=True, 
            padding="max_length"
        )
        
        inputs["labels"] = targets["input_ids"]
        return inputs

    tokenized_train = train_data.map(lambda x: tokenize_function(x, use_teacher=True), batched=True)
    tokenized_val = val_data.map(lambda x: tokenize_function(x, use_teacher=False), batched=True)

    args = Seq2SeqTrainingArguments(
        output_dir="./t5_distilled_student",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        predict_with_generate=True,
        bf16=torch.cuda.is_available()
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model("./flan_t5_distilled_student_self_instruct/final")

if __name__ == "__main__":
    train_student()