# train_embedding.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
train_examples = [InputExample(texts=["Tesla makes electric cars.", "Tesla produces EVs."], label=1.0)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=1)
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1
)

os.makedirs("models/fine_tuned_embedding", exist_ok=True)
model.save("models/fine_tuned_embedding")
print("Fine-tuned embedding model saved!")