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

def preprocess_self_instruct_dataset(dataset):
  """Prepare Self-Instruct examples for teacher generation and training."""

  def preprocess_sample(sample):
      prompt = sample.get('prompt', '').strip()

      sample['input'] = f'instruction: {prompt}'
      sample['input'] = ' '.join(sample['input'].split())

      sample['output'] = sample.get('completion', '').strip()

      return sample

  dataset = dataset.map(preprocess_sample)
  dataset = dataset.remove_columns(
      ['prompt', 'completion']
  )

  TRAIN_WEIGHT = 56000
  VALIDATION_WEIGHT = 8000
  TEST_WEIGHT = 16000

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

  dataset = datasets.DatasetDict({
      'train': dataset_split_0['train'],
      'validation': dataset_split_1['train'],
      'test': dataset_split_1['test']
  })

  return dataset

if __name__ == "__main__":
    raw_dataset = datasets.load_dataset(
        "parquet",
        data_files="hf://datasets/yizhongw/self_instruct@refs/convert/parquet/self_instruct/**/*.parquet"
    )
    processed_dataset = preprocess_self_instruct_dataset(raw_dataset)

    processed_dataset.save_to_disk(str(PROCESSED_DATA_DIR / "self_instruct_splits"))
