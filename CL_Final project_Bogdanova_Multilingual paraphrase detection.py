!pip install transformers
!pip install datasets
!pip install evaluate
!pip install bertviz transformers
!pip install accelerate --upgrade

from google.colab import drive
drive.mount('/content/drive')

from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-base")
print(tokenizer)

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorWithPadding

ds_de = load_dataset("google-research-datasets/paws-x", "de")
ds_en = load_dataset("google-research-datasets/paws-x", "en")

print(ds_de)
print(ds_en)

# In order to do multilungual and initially cross-linguag (on the test) paraphrase detection,
# it is necessary to concatenate English and German data sets

from datasets import concatenate_datasets # Concatination step just stack data sets on top of each other

# Concatenate the 'train' splits from English and German
train_combined = concatenate_datasets([ds_en['train'], ds_de['train']])

# Concatenate the 'validation' splits
validation_combined = concatenate_datasets([ds_en['validation'], ds_de['validation']])

# Concatenate the 'test' splits
test_combined = concatenate_datasets([ds_en['test'], ds_de['test']])

# Shuffle each split so that examples from EN and DE are mixed
train_combined = train_combined.shuffle(seed=42)
validation_combined = validation_combined.shuffle(seed=42)
test_combined = test_combined.shuffle(seed=42)

# Store everything in a single DatasetDict
ds_combined = DatasetDict({
    'train': train_combined,
    'validation': validation_combined,
    'test': test_combined
})

print(ds_combined)


# Subset the dataset to 500 training examples, 200 for validation/test

ds_combined['train'] = ds_combined['train'].shuffle(seed=42).select(range(500))
ds_combined['validation'] = ds_combined['validation'].shuffle(seed=42).select(range(200))
ds_combined['test'] = ds_combined['test'].shuffle(seed=42).select(range(200))

# Print out the first 10 examples from the train dataset to verify correctness
ds_combined["train"][:10]

# We need to preprocess the data for training

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding='max_length',
        truncation=True,
        max_length=128 # Token leght for truncation and handling too long sentences
    )

small_tokenized_dataset = ds_combined.map(tokenize_function, batched=True)

# rename 'label' to 'labels' > better for BERT like models that expect this namimg
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")

# set format for PyTorch
small_tokenized_dataset.set_format("torch")


# Check what the first two sequences look like

small_tokenized_dataset['train'][0:2]

# Specifyng all the trainig parameters
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import set_seed

set_seed(123)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
accuracy = evaluate.load("accuracy")


arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/finalproject/paraphrase_detection_1st",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=8,
    num_train_epochs=5,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to='none',
    seed=123
)

def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['validation'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Specifyng all the trainig parameters
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import set_seed

set_seed(123)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
accuracy = evaluate.load("accuracy")


arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/finalproject/paraphrase_detection_1st",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=8,
    num_train_epochs=5,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,train_combined = concatenate_datasets([ds_en['train'], ds_de['train']])

validation_combined = concatenate_datasets([ds_en['validation'], ds_de['validation']])

test_combined = concatenate_datasets([ds_en['test'], ds_de['test']])

train_combined = train_combined.shuffle(seed=42)
validation_combined = validation_combined.shuffle(seed=42)
test_combined = test_combined.shuffle(seed=42)

ds_combined = DatasetDict({
    'train': train_combined,
    'validation': validation_combined,
    'test': test_combined
})

print(ds_combined)


ds_combined['train'] = ds_combined['train'].shuffle(seed=42).select(range(2000))
ds_combined['validation'] = ds_combined['validation'].shuffle(seed=42).select(range(500))
ds_combined['test'] = ds_combined['test'].shuffle(seed=42).select(range(500))

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding='max_length',
        truncation=True,
        max_length=150
    )

small_tokenized_dataset = ds_combined.map(tokenize_function, batched=True)
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")
    load_best_model_at_end=True,
    report_to='none',
    seed=123
)

def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['validation'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.train()

# Check for the label distribution in the train data set due to the poor accuracy of <61%
from collections import Counter
count_0 = 0
count_1 = 0

for label in small_tokenized_dataset["train"]["labels"]:
    if label == 0:
        count_0 += 1
    elif label == 1:
        count_1 += 1

print(f"Label 0: {count_0}")
print(f"Label 1: {count_1}")

# 2d Training

train_combined = concatenate_datasets([ds_en['train'], ds_de['train']])

validation_combined = concatenate_datasets([ds_en['validation'], ds_de['validation']])

test_combined = concatenate_datasets([ds_en['test'], ds_de['test']])

train_combined = train_combined.shuffle(seed=42)
validation_combined = validation_combined.shuffle(seed=42)
test_combined = test_combined.shuffle(seed=42)

ds_combined = DatasetDict({
    'train': train_combined,
    'validation': validation_combined,
    'test': test_combined
})

print(ds_combined)


ds_combined['train'] = ds_combined['train'].shuffle(seed=42).select(range(2000))
ds_combined['validation'] = ds_combined['validation'].shuffle(seed=42).select(range(500))
ds_combined['test'] = ds_combined['test'].shuffle(seed=42).select(range(500))

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding='max_length',
        truncation=True,
        max_length=150
    )

small_tokenized_dataset = ds_combined.map(tokenize_function, batched=True)
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")


print("Train size:", ds_combined["train"].num_rows)
print("Validation size:", ds_combined["validation"].num_rows)
print("Test size:", ds_combined["test"].num_rows)

from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import set_seed

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
accuracy = evaluate.load("accuracy")

set_seed(24)

arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/finalproject/paraphrase_detection_2d",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_steps=8,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to='none',
    seed=24
)

def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)


trainer.train()

# Model evaluation

results = trainer.predict(small_tokenized_dataset['validation'])
print("Validation results:", results.metrics)

results = trainer.predict(small_tokenized_dataset["test"])
print("Test results:", results.metrics)

# 3d Training

train_combined = concatenate_datasets([ds_en['train'], ds_de['train']])

validation_combined = concatenate_datasets([ds_en['validation'], ds_de['validation']])

test_combined = concatenate_datasets([ds_en['test'], ds_de['test']])

train_combined = train_combined.shuffle(seed=42)
validation_combined = validation_combined.shuffle(seed=42)
test_combined = test_combined.shuffle(seed=42)

ds_combined = DatasetDict({
    'train': train_combined,
    'validation': validation_combined,
    'test': test_combined
})

print(ds_combined)


ds_combined['train'] = ds_combined['train'].shuffle(seed=42).select(range(3000))
ds_combined['validation'] = ds_combined['validation'].shuffle(seed=42).select(range(500))
ds_combined['test'] = ds_combined['test'].shuffle(seed=42).select(range(500))

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding='max_length',
        truncation=True,
        max_length=150
    )

small_tokenized_dataset = ds_combined.map(tokenize_function, batched=True)
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")

from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import set_seed

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)
accuracy = evaluate.load("accuracy")

set_seed(34)

arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/finalproject/paraphrase_detection_3d",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=8,
    num_train_epochs=8,
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to='none',
    seed=34
)

def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return accuracy.compute(predictions=predictions, references=labels)

# from transformers import EarlyStoppingCallback > I had to comment it out as the training stopped after 6th epoch

# early_stopping = EarlyStoppingCallback(
#     early_stopping_patience=2,  # Stop after 'patience' epochs with no improvement
#     early_stopping_threshold=0.0  # How much improvement must be seen
#)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['validation'], # change to test when you do your final evaluation!
    tokenizer=tokenizer,
    data_collator=data_collator,
    #callbacks=[early_stopping],
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.predict(small_tokenized_dataset["test"])
print("Test results:", results.metrics)

# Searching for best HP with grid search

import itertools
import numpy as np
from transformers import TrainingArguments, Trainer, set_seed, AutoModelForSequenceClassification
import evaluate
from transformers import DataCollatorWithPadding

train_combined = concatenate_datasets([ds_en['train'], ds_de['train']])

validation_combined = concatenate_datasets([ds_en['validation'], ds_de['validation']])

test_combined = concatenate_datasets([ds_en['test'], ds_de['test']])

train_combined = train_combined.shuffle(seed=42)
validation_combined = validation_combined.shuffle(seed=42)
test_combined = test_combined.shuffle(seed=42)

ds_combined = DatasetDict({
    'train': train_combined,
    'validation': validation_combined,
    'test': test_combined
})

print(ds_combined)


ds_combined['train'] = ds_combined['train'].shuffle(seed=42).select(range(2000))
ds_combined['validation'] = ds_combined['validation'].shuffle(seed=42).select(range(500))
ds_combined['test'] = ds_combined['test'].shuffle(seed=42).select(range(500))

def tokenize_function(examples):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding='max_length',
        truncation=True,
        max_length=150
    )

small_tokenized_dataset = ds_combined.map(tokenize_function, batched=True)
small_tokenized_dataset = small_tokenized_dataset.rename_column("label", "labels")
small_tokenized_dataset.set_format("torch")

# HP grid
learning_rates = [2e-5, 3e-5]
weight_decays = [0.0, 0.01]
batch_sizes = [16, 32]
num_epochs_list = [5, 8]

accuracy = evaluate.load("accuracy")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


grid_search_results = []
grid_counter = 0

for lr in learning_rates:
    for wd in weight_decays:
        for bs in batch_sizes:
            for num_epochs in num_epochs_list:
                grid_counter += 1
                print(f"\n--- Grid Search Iteration {grid_counter} ---")
                print(f"Learning Rate: {lr}, Weight Decay: {wd}, Batch Size: {bs}, Epochs: {num_epochs}")

                output_dir = f"/content/drive/MyDrive/finalproject/grid_search/model_{grid_counter}"

                set_seed(42)

                model = AutoModelForSequenceClassification.from_pretrained("xlm-roberta-base", num_labels=2)

                training_args = TrainingArguments(
                    output_dir=output_dir,
                    per_device_train_batch_size=bs,
                    per_device_eval_batch_size=bs,
                    logging_steps=8,
                    num_train_epochs=num_epochs,
                    eval_strategy="epoch",  # Evaluate at the end of each epoch
                    save_strategy="epoch",
                    learning_rate=lr,
                    weight_decay=wd,
                    load_best_model_at_end=True,
                    report_to='none',
                    seed=42
                )

                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    return accuracy.compute(predictions=predictions, references=labels)

                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=small_tokenized_dataset['train'],
                    eval_dataset=small_tokenized_dataset['validation'],
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=compute_metrics
                )

                trainer.train()

                eval_result = trainer.evaluate()
                print(f"Validation Evaluation Results: {eval_result}")

                grid_search_results.append({
                    "lr": lr,
                    "weight_decay": wd,
                    "batch_size": bs,
                    "num_epochs": num_epochs,
                    "eval_accuracy": eval_result.get("eval_accuracy", None)
                })


for result in grid_search_results:
    print(result)


# Evaluation

import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
mcc_metric = evaluate.load("matthews_correlation")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-564")
tokenizer_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

eval_dataloader = DataLoader(small_tokenized_dataset['test'], batch_size=8)

fine_tuned_model.eval()

for batch in eval_dataloader:
    # Tokenize using the two sentence columns from your dataset
    inputs = tokenizer(
        batch['sentence1'],
        batch['sentence2'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )


    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    accuracy_metric.add_batch(predictions=predictions, references=batch['labels'])
    f1_metric.add_batch(predictions=predictions, references=batch['labels'])
    mcc_metric.add_batch(predictions=predictions, references=batch['labels'])
    precision_metric.add_batch(predictions=predictions, references=batch['labels'])
    recall_metric.add_batch(predictions=predictions, references=batch['labels'])

results = {
    "accuracy": accuracy_metric.compute(),
    "f1": f1_metric.compute(average="binary"),
    "matthews_correlation": mcc_metric.compute(),
    "precision": precision_metric.compute(average="binary"),
    "recall": recall_metric.compute(average="binary")
}

print(results)

checkpoint_path = "/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-564"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
model.eval()

tokenizer_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

label_names = ["NO", "YES"]


test_pairs = [
    ("I love cats", "Ich liebe Katzen", "YES"),  # paraphrase
    ("Where is the library?", "Wo ist die Bibliothek?", "YES"),
    ("She is very happy", "Sie ist sehr glücklich", "YES"),
    ("The weather is nice today", "Heute ist das Wetter schön", "YES"),
    ("I enjoy reading books", "Ich lese gerne Bücher", "YES"),
    ("Can you help me?", "Kannst du mir bitte helfen?", "YES"),
    ("They are going to the cinema", "Sie gehen ins Kino", "YES"),
    ("He found a new job", "Er hat seinen Job gekündigt", "NO"), # not paraphrase
    ("He bought a new car", "Er hat sein Fahrrad verkauft", "NO"),
    ("The meeting starts at 3 PM", "Das Meeting wurde abgesagt", "NO"),
    ("She likes coffee", "Er trinkt Tee", "NO"),
    ("We are traveling to France", "Wir waren letztes Jahr in Spanien", "NO"),
    ("My brother is a doctor", "Meine Schwester studiert Kunst", "NO")
]

correct_predictions = 0

for idx, (sent1, sent2, expected) in enumerate(test_pairs):
    model_inputs = tokenizer(sent1, sent2, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits

    predicted_label_id = torch.argmax(logits, dim=-1).item()
    predicted_label_name = label_names[predicted_label_id]

    print(f"Example {idx+1}:")
    print(f"Sentence 1: {sent1}")
    print(f"Sentence 2: {sent2}")
    print(f"Predicted label: {predicted_label_name}  (Expected: {expected})\n")

    if predicted_label_name == expected:
        correct_predictions += 1

accuracy = correct_predictions / len(test_pairs)
print(accuracy)

# Confusion matrix

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-564")
tokenizer_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

eval_dataloader = DataLoader(small_tokenized_dataset['test'], batch_size=8)

fine_tuned_model.eval()

all_predictions = []
all_true_labels = []

for batch in eval_dataloader:
    inputs = tokenizer(
        batch['sentence1'],
        batch['sentence2'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )


    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

    all_predictions.extend(predictions.cpu().numpy())
    all_true_labels.extend(batch['labels'] if isinstance(batch['labels'], list) else batch['labels'].cpu().numpy())

cm = confusion_matrix(all_true_labels, all_predictions)
print("Confusion Matrix:\n", cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Paraphrase", "Paraphrase"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix on Test Set")
plt.show()

# In order to better inspect the false positive and false negative examples, I would like to take a look at them

import pandas as pd

error_examples = []

for i in range(len(small_tokenized_dataset['test'])):
    example = small_tokenized_dataset['test'][i]

    inputs = tokenizer(
        example['sentence1'],
        example['sentence2'],
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    true_label = example['labels']

    if prediction != true_label:
        error_examples.append({
            'id': example['id'],
            'sentence1': example['sentence1'],
            'sentence2': example['sentence2'],
            'true_label': true_label,
            'predicted_label': prediction,
            'sentence1_length': len(example['sentence1']),
            'sentence2_length': len(example['sentence2'])
        })


df_errors = pd.DataFrame(error_examples)
display(df_errors)


!pip install bertviz
!pip install bertviz transformers

from torch.utils.tensorboard import SummaryWriter
import re
import tensorflow as tf
import tensorboard as tb

from transformers import AutoModelForSequenceClassification

# Model loading for specific check points (epoch 6) - the best model

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-564"
)
#model_inputs = tokenizer(small_tokenized_dataset["test"]['sentence1']["sentence2"], padding=True, truncation=True, return_tensors="pt")
#print(len(small_tokenized_dataset["test"]))
num_examples = 100

small_test_dataset = small_tokenized_dataset["test"].shuffle(seed=42).select(range(num_examples))

sentence1_list = small_test_dataset["sentence1"]
sentence2_list = small_test_dataset["sentence2"]
labels_list    = small_test_dataset["labels"]

model_inputs = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    max_length=150,
    return_tensors='pt'
)

outputs = fine_tuned_model(**model_inputs, output_hidden_states=True)


import torch
import os

all_hidden_states = outputs.hidden_states
num_layers = len(all_hidden_states)

path = "result_viz_roberta_1"


for layer in range(num_layers):
    layer_dir = os.path.join(path, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)

    tensors = []
    labels = []

    for example in range(all_hidden_states[layer].shape[0]):
        cls_embedding = all_hidden_states[layer][example][0]
        tensors.append(cls_embedding)

        label = [
            sentence1_list[example],
            str(labels_list[example])
        ]
        labels.append(label)

    embeddings_tensor = torch.stack(tensors)

    writer = SummaryWriter(log_dir=layer_dir)
    writer.add_embedding(mat=embeddings_tensor, metadata=labels, metadata_header=['text', 'label'])




# Model loading for specific check points (epoch 2)

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-188"
)
#model_inputs = tokenizer(small_tokenized_dataset["test"]['sentence1']["sentence2"], padding=True, truncation=True, return_tensors="pt")
#print(len(small_tokenized_dataset["test"]))
num_examples = 100

small_test_dataset = small_tokenized_dataset["test"].shuffle(seed=42).select(range(num_examples))

sentence1_list = small_test_dataset["sentence1"]
sentence2_list = small_test_dataset["sentence2"]
labels_list    = small_test_dataset["labels"]

model_inputs = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    max_length=150,
    return_tensors='pt'
)

outputs = fine_tuned_model(**model_inputs, output_hidden_states=True)


import torch
import os

all_hidden_states = outputs.hidden_states
num_layers = len(all_hidden_states)

path = "result_viz_roberta_2"


for layer in range(num_layers):
    layer_dir = os.path.join(path, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)

    tensors = []
    labels = []

    for example in range(all_hidden_states[layer].shape[0]):
        cls_embedding = all_hidden_states[layer][example][0]
        tensors.append(cls_embedding)

        label = [
            sentence1_list[example],
            str(labels_list[example])
        ]
        labels.append(label)

    embeddings_tensor = torch.stack(tensors)

    writer = SummaryWriter(log_dir=layer_dir)
    writer.add_embedding(mat=embeddings_tensor, metadata=labels, metadata_header=['text', 'label'])



# Model loading for specific check points (epoch 3)

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(
    "/content/drive/MyDrive/finalproject/paraphrase_detection_3d/checkpoint-282"
)
#model_inputs = tokenizer(small_tokenized_dataset["test"]['sentence1']["sentence2"], padding=True, truncation=True, return_tensors="pt")
#print(len(small_tokenized_dataset["test"]))
num_examples = 100

small_test_dataset = small_tokenized_dataset["test"].shuffle(seed=42).select(range(num_examples))

sentence1_list = small_test_dataset["sentence1"]
sentence2_list = small_test_dataset["sentence2"]
labels_list    = small_test_dataset["labels"]

model_inputs = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    max_length=150,
    return_tensors='pt'
)

outputs = fine_tuned_model(**model_inputs, output_hidden_states=True)



import torch
import os

all_hidden_states = outputs.hidden_states
num_layers = len(all_hidden_states)

path = "result_viz_roberta_3"


for layer in range(num_layers):
    layer_dir = os.path.join(path, f"layer_{layer}")
    os.makedirs(layer_dir, exist_ok=True)

    tensors = []
    labels = []

    for example in range(all_hidden_states[layer].shape[0]):
        cls_embedding = all_hidden_states[layer][example][0]
        tensors.append(cls_embedding)

        label = [
            sentence1_list[example],
            str(labels_list[example])
        ]
        labels.append(label)

    embeddings_tensor = torch.stack(tensors)

    writer = SummaryWriter(log_dir=layer_dir)
    writer.add_embedding(mat=embeddings_tensor, metadata=labels, metadata_header=['text', 'label'])
