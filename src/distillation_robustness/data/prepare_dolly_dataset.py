import random

import datasets
import numpy
import torch

from distillation_robustness.paths import PROCESSED_DATA_DIR

SEED = 42
DEVICE = 'cpu'
if torch.cuda.is_available():
  DEVICE = 'cuda'
print(f'Device: {DEVICE}')

SEED = 42

random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def preprocess_databricks_dolly_15k_dataset(dataset):
    """Prepare Dolly examples for encoder-decoder instruction tuning."""

    def filter_sample(sample):
        if sample['category'] == 'classification':
            return sample['context'] != ''
        return sample['category'] in ['closed_qa', 'information_extraction', 'summarization']

    def preprocess_sample(sample):
        instruction = sample.get('instruction', '').strip()
        context = sample.get('context', '').strip()
        category = sample.get('category', '')

        if category in ['closed_qa', 'information_extraction']:
            sample['input'] = f'question: {instruction} context: {context}'
        elif category == 'summarization':
            text_to_summarize = f'{instruction} {context}'.strip()
            sample['input'] = f'summarize: {text_to_summarize}'
        elif category == 'classification':
            if context:
                sample['input'] = f'question: {instruction} context: {context}'
            else:
                sample['input'] = instruction

        sample['input'] = ' '.join(sample['input'].split())
        sample['output'] = sample.get('response', '').strip()

        return sample

    dataset = dataset.filter(filter_sample)
    dataset = dataset.map(preprocess_sample)

    dataset = dataset.remove_columns(
        ['instruction', 'context', 'response', 'category']
    )

    TRAIN_WEIGHT = 10
    VALIDATION_WEIGHT = 1
    TEST_WEIGHT = 4

    dataset_split_0 = dataset['train'].train_test_split(
        train_size=TRAIN_WEIGHT / (TRAIN_WEIGHT + VALIDATION_WEIGHT + TEST_WEIGHT),
        test_size=(VALIDATION_WEIGHT + TEST_WEIGHT) / (TRAIN_WEIGHT + VALIDATION_WEIGHT + TEST_WEIGHT),
        seed=SEED
    )
    dataset_split_1 = dataset_split_0['test'].train_test_split(
        train_size=VALIDATION_WEIGHT / (VALIDATION_WEIGHT + TEST_WEIGHT),
        test_size=TEST_WEIGHT / (VALIDATION_WEIGHT + TEST_WEIGHT),
        seed=SEED
    )

    return datasets.DatasetDict({
        'train': dataset_split_0['train'],
        'validation': dataset_split_1['train'],
        'test': dataset_split_1['test']
    })

if __name__ == "__main__":
    raw_dataset = datasets.load_dataset("databricks/databricks-dolly-15k")
    processed_dataset = preprocess_databricks_dolly_15k_dataset(raw_dataset)

    processed_dataset.save_to_disk(str(PROCESSED_DATA_DIR / "dolly_splits"))
