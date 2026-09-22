# Data

The dataset used for the recorded experiments is not included in this repository.

The original data contain images paired with four-option visual questions. Validation and test questions also have caption IDs that can be matched back to a shared image folder.

The VQA code expects JSONL records with fields like:

```json
{
  "image": "images/test_val/001313385.jpg",
  "question": "What is the main purpose of the image?",
  "options": ["option A", "option B", "option C", "option D"],
  "answer": "A",
  "caption_id": "val_0"
}
```

For image recovery, caption files are expected to contain:

```json
{
  "caption_id": "val_0",
  "caption": "A description of the image"
}
```

The source material describes the image collection as an in-house corpus and does not provide redistribution terms, so the dataset and images are deliberately omitted here.
