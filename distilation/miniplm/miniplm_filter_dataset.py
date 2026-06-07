from datasets import load_from_disk

def filter_dataset():
    dataset = load_from_disk("./miniplm/self_instruct_train_miniplm_scored")
    
    def compute_diff(sample):
        sample['difference_score'] = sample['teacher_score'] - sample['ref_score']
        return sample
        
    dataset = dataset.map(compute_diff)
    
    sorted_dataset = dataset.sort("difference_score", reverse=True)
    
    KEEP_RATIO = 0.60
    keep_count = int(len(sorted_dataset) * KEEP_RATIO)
    
    filtered_dataset = sorted_dataset.select(range(keep_count))
    
    print("-" * 30)
    print(f"Original Dataset Size: {len(dataset)}")
    print(f"Filtered Dataset Size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} redundant/noisy samples.")
    print("-" * 30)
    
    filtered_dataset.save_to_disk("./miniplm/self_instruct_train_miniplm_filtered")
    print("Success! Final curated MiniPLM dataset saved.")

if __name__ == "__main__":
    filter_dataset()