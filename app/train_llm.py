# train_llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load Q&A pairs from your data
qa_pairs = []
with open("data/metadata.jsonl", "r") as f:
    docs = [json.loads(line) for line in f]

for doc in docs:
    qa_pairs.append({
        "instruction": "Answer using only the context.",
        "input": f"Context: {doc['text']}\nQuestion: What is mentioned?",
        "output": doc["text"][:200]  # Simulate answer
    })

dataset = Dataset.from_list(qa_pairs)

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM3-3B")

def tokenize_function(examples):
    inputs = [f"{i}\n{o}" for i, o in zip(examples["input"], examples["output"])]
    return tokenizer(inputs, truncation=True, max_length=512, padding="max_length")

tokenized = dataset.map(tokenize_function, batched=True)

args = TrainingArguments(
    output_dir="models/fine_tuned_llm",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_steps=500,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()

model.save_pretrained("models/fine_tuned_llm")
tokenizer.save_pretrained("models/fine_tuned_llm")