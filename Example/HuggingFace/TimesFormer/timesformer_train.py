import torch
from transformers import TimesformerForVideoClassification, AutoImageProcessor
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from PIL import Image
import cv2
import evaluate

# Ładowanie zbioru danych
dataset = load_dataset('video_folder', data_dir='path_to_your_dataset')

# Ładowanie modelu i procesora
model_name = 'facebook/timesformer-base-finetuned-k400'
model = TimesformerForVideoClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Funkcja do ekstrakcji klatek
def extract_frames(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
    cap.release()
    return frames

# Funkcja preprocessująca
def preprocess_function(examples):
    video_paths = examples['video_path']
    labels = examples['label']
    frames = [extract_frames(video_path) for video_path in video_paths]
    inputs = processor(frames, return_tensors="pt", padding="max_length", truncation=True)
    inputs['labels'] = labels
    return inputs

# Zastosowanie preprocessingu
tokenized_datasets = dataset.map(preprocess_function, batched=True, batch_size=4)

# Definicja metryk
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Konfiguracja argumentów treningowych
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
)

# Inicjalizacja Trenera
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['val'],
    compute_metrics=compute_metrics,
)

# Trening
trainer.train()

# Ewaluacja
results = trainer.evaluate()
print(results)

# Zapisanie modelu
trainer.save_model("path_to_save_your_model")
processor.save_pretrained("path_to_save_your_processor")
