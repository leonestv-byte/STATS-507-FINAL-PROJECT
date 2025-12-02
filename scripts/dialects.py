from datasets import load_dataset
from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
import evaluate
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch.nn as nn
from transformers import PretrainedConfig
from datasets import Dataset
import matplotlib.pyplot as plt
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModelForSeq2SeqLM
import evaluate
import torch

##################################################
#########   Load the DS Appropriately    #########
##################################################

egy_ds = load_dataset("QCRI/arabic_pos_dialect", "egy")
glf_ds = load_dataset("QCRI/arabic_pos_dialect", "glf")
lev_ds = load_dataset("QCRI/arabic_pos_dialect", "lev")
mgr_ds = load_dataset("QCRI/arabic_pos_dialect", "mgr")

dialect_ds = egy_ds

#####################################
######  ID To Label Mapping #########
#####################################

id2label = []


######################################
#########   Graph Results    #########
######################################

def graph(stats):
	labels = list(stats.keys())
	values = list(stats.values())
	plt.figure(figsize=(10, 5))
	bars = plt.bar(labels, values)

	# Add labels on top of each bar
	for bar, value in zip(bars, values):
		height = bar.get_height()
		plt.text(
			bar.get_x() + bar.get_width() / 2,
			height,
			f"{value:.3f}",
			ha='center',
			va='bottom'
		)

	plt.xticks(rotation=45, ha='right')
	plt.ylabel("Value")
	plt.title("Evaluation Statistics")
	plt.tight_layout()
	plt.show()


model_name = "UBC-NLP/AraT5v2-base-1024"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128

label2id = {label:i for i,label in enumerate(id2label)}


# We take an example (segments, pos_tags)
def preprocess_example(example):
    model_inputs = tokenizer(
        example["segments"],
        max_length=max_length,
        truncation=True,
        is_split_into_words=True
    )

    # Fill in all parts of speech in the dataset.
    for tag in example["pos_tags"]:
        if label2id.get(tag, -1) == -1:
            label2id[tag] = len(label2id) + 1

    labels = [ label2id.get(tag) for tag in example["pos_tags"] ]
    
    # pad model_inputs, labels, attn_mask
    input_ids = np.pad(model_inputs["input_ids"],
                       (0, max_length - len(model_inputs["input_ids"])),
                       mode='constant')
    label_ids = np.pad(labels,
                       (0, max_length - len(labels)),
                       mode='constant')
    attn_mask = np.pad(model_inputs["attention_mask"],
                       (0, max_length - len(model_inputs["attention_mask"])),
                       mode='constant')

    # Return it in dictionary forms
    return {
        "input_ids": input_ids,
        "labels": label_ids,
        "attention_mask": attn_mask
    }

rows = [preprocess_example(ex) for ex in dialect_ds["train"]]
df = pd.DataFrame(rows)
ds = Dataset.from_pandas(df.copy())


def fix_dataset(batch):
    input_ids = []
    labels = []
    attention_mask = []

    for i in range(len(batch["input_ids"])):
        # input_ids
        ids = batch["input_ids"][i][:max_length]
        ids += [0] * (max_length- len(ids))
        input_ids.append(ids)

        # labels
        lab = batch["labels"][i][:max_length]
        lab += [0] * (max_length- len(lab))
        labels.append(lab)

        # attention_mask
        mask = batch["attention_mask"][i][:max_length]
        mask += [0] * (max_length - len(mask))
        attention_mask.append(mask)

    # Convert to tensors
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }

id2label = list(label2id.keys())
num_pos_tags = len(label2id)


#######################################
#########   Compute Metrcs    #########
#######################################


accuracy_metric = evaluate.load("accuracy")

# Compute Metrics - 1 (pretrained model)
def compute_metrics(eval_pred):
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    new_predictions = []
    new_label_ids = []
    for prediction in predictions:
        new_prediction = np.pad(prediction, (0, max_length - len(prediction)), mode='constant')
        new_predictions.append(new_prediction)
    for label_id in label_ids:
    	new_label_id = np.pad(label_id, (0, max_length - len(label_id)), mode='constant')
    	new_label_ids.append(new_label_id)
    new_predictions = np.array(new_predictions)
    new_label_ids = np.array(new_label_ids)
    return accuracy_metric.compute(predictions=new_predictions.flatten(), references=new_label_ids.flatten())


# Compute metrics 2 (Custom Architecture - Simple Linear Layer)
def compute_metrics_two(eval_pred):
    logits, labels = eval_pred
    
    # Convert logits to token ids
    predictions = np.argmax(logits, axis=-1)

    # Ensure same shape: pad or truncate predictions to match labels
    max_len = labels.shape[1]   # labels are already padded by data collator
    
    # Pad/truncate predictions
    if predictions.shape[1] < max_len:
        pad_width = max_len - predictions.shape[1]
        predictions = np.pad(predictions, ((0,0),(0,pad_width)), constant_values=-100)
    else:
        predictions = predictions[:, :max_len]

    # Flatten only AFTER shapes match
    return accuracy_metric.compute(
        predictions=predictions.flatten(),
        references=labels.flatten()
    )

# Compute metrics 3 (Custom Architecture - BiLSTM)
def compute_metrics_three(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    preds_flat = preds.flatten()
    labels_flat = labels.flatten()

    mask = labels_flat != -100
    preds_flat = preds_flat[mask]
    labels_flat = labels_flat[mask]

    return accuracy_metric.compute(predictions=preds_flat, references=labels_flat)


#########################################
#########   Pretrained Model    #########
#########################################

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


training_args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_output",           
    num_train_epochs=1,                      
    per_device_train_batch_size=2,           
    per_device_eval_batch_size=2,            
    learning_rate=2e-5,                      
    weight_decay=0.01,                       
    logging_dir="./logs",                    
    predict_with_generate=True,              
    evaluation_strategy="steps",             
    eval_steps=500,
    use_mps_device=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    compute_metrics=compute_metrics
)

# Training Results
# trainer.train()
# stats = trainer.evaluate()
# print(stats)
# graph(stats)


# Test Results
# Custom split (70/30)
split_dataset = ds.train_test_split(test_size=0.3, seed=42) # Seed for reproducibility

# Access the resulting splits
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
stats = trainer.evaluate(test_dataset)
print(stats)
graph(stats)


##################################################
#########   Embeddings + Linear Model    #########
##################################################


# Vocab size.
# There are ~44800 vocabulary words in arabic.
vocab_size = 44800

# Output Shape
output_shape = num_pos_tags

# This is our custom Sequence2Sequence model. 
#
# The embeddings layers is first - there is
# no matric multiplication, but it maps our
# R^(1x128) input_ids to a R^(128x206) embeddings matrix
# (if we have up to 128 tokens, embeddings up to 206 for the hidden layer).
#
# The linear layer is a typical linear layer.
# W is a matrix R^(206x44800)
# b is a bias term, R^(44800).
# output = xW + b, R^(128x44800)
#
class MySeq2SeqModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(vocab_size, 206)
        self.linear = nn.Linear(206, output_shape)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.linear(x)
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, output_shape),
                labels.view(-1)
            )
        return Seq2SeqLMOutput(logits=logits, loss=loss)

ds = Dataset.from_pandas(df.copy())
ds = ds.map(fix_dataset, batched=True)
ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

config = PretrainedConfig()
model = MySeq2SeqModel(config)

# Define our training args.
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",            # where to save checkpoints
    overwrite_output_dir=True,         # overwrite existing outputs
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=16,    # adjust based on your GPU memory
    per_device_eval_batch_size=16,     
    learning_rate=2e-5,                # decent starting LR for small models
    weight_decay=0.01,                 # regularization
    save_total_limit=2,                # keep last 2 checkpoints
    num_train_epochs=100, #1               # can increase if dataset is small
    logging_dir="./logs",              # logs for tensorboard
    logging_steps=10,                  # log every 10 steps
    save_steps=50,                     # checkpoint frequency
    evaluation_strategy="steps",       # evaluate during training
    eval_steps=50,                     # same as save_steps
    predict_with_generate=False,       # our model is simple LM; no generation needed
    dataloader_num_workers=0,          # prevent freezing/deadlocks
    report_to="none",                  # disable WandB or other logging for simplicity
    fp16=False                          # enable if using GPU and want speed
)


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    compute_metrics=compute_metrics_two
)

# Training Only Results
# trainer.train()
# stats = trainer.evaluate(ds)
# print(stats)
# graph(stats)


# Test Results
split_dataset = ds.train_test_split(test_size=0.3, seed=42)

# Access the resulting splits
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics_two
)

# Test Results
trainer.train()
stats = trainer.evaluate(test_dataset)
print(stats)
graph(stats)



###########################################
#########   BiLSTM Custom Model   #########
###########################################

# Output Shape
output_shape = num_pos_tags


class BiLSTMCustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(vocab_size, 206)
        self.bilstm = nn.LSTM(206, 200, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(400, output_shape)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        lstm_out, _ = self.bilstm(x)
        logits = self.linear(lstm_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, output_shape), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
            )



training_args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_output",           # Required: directory for saving model checkpoints
    num_train_epochs=20,                      # Total number of training epochs
    per_device_train_batch_size=2,           # Batch size per device during training
    per_device_eval_batch_size=2,            # Batch size for evaluation
    learning_rate=2e-5,                      # The initial learning rate
    weight_decay=0.01,                       # Strength of weight decay
    logging_dir="./logs",                    # Directory for storing logs
    predict_with_generate=False,              # Use generate to compute the evaluation metrics
    evaluation_strategy="steps",             # Evaluate every `eval_steps`
    eval_steps=500,
    use_mps_device=False
)

config = PretrainedConfig()
model = BiLSTMCustomModel(config)

# Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    compute_metrics=compute_metrics_three
)

# trainer.train()
# stats = trainer.evaluate(test_dataset)
# print(stats)
# graph(stats)

# Test Results
split_dataset = ds.train_test_split(test_size=0.3, seed=42) # Seed for reproducibility
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics_three
)

trainer.train()
stats = trainer.evaluate(test_dataset)
print(stats)
graph(stats)

###########################################
#########   Encoder Decoder       #########
###########################################

training_args = Seq2SeqTrainingArguments(
    output_dir="./seq2seq_output",           # Required: directory for saving model checkpoints
    num_train_epochs=1,                      # Total number of training epochs
    per_device_train_batch_size=2,           # Batch size per device during training
    per_device_eval_batch_size=2,            # Batch size for evaluation
    learning_rate=2e-5,                      # The initial learning rate
    weight_decay=0.01,                       # Strength of weight decay
    logging_dir="./logs",                    # Directory for storing logs
    predict_with_generate=False,              # Use generate to compute the evaluation metrics
    evaluation_strategy="steps",             # Evaluate every `eval_steps`
    eval_steps=500,
    use_mps_device=False
)


# Output Shape
output_shape = num_pos_tags

class EncoderDecoderModel(PreTrainedModel):
    def __init__(self, config, vocab_size=vocab_size, hidden_size=256, nhead=8, num_layers=2):
        super().__init__(config)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Embeddings
        self.encoder_embed = nn.Embedding(vocab_size, hidden_size)
        self.decoder_embed = nn.Embedding(vocab_size, hidden_size)
        
        # Encoder & Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.linear = nn.Linear(hidden_size, output_shape)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None):
        # Encoder
        enc_emb = self.encoder_embed(input_ids)  # [batch, seq, hidden]
        enc_emb = enc_emb.transpose(0, 1)        # Transformer expects [seq, batch, hidden]
        memory = self.encoder(enc_emb)           # [seq, batch, hidden]
        
        # Decoder
        dec_emb = self.decoder_embed(labels)
        dec_emb = dec_emb.transpose(0, 1)
        # Generate tgt_mask for causal decoding
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(dec_emb.size(0)).to(dec_emb.device)
        output = self.decoder(dec_emb, memory, tgt_mask=tgt_mask)
        
        output = output.transpose(0, 1)  # back to [batch, seq, hidden]
        logits = self.linear(output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, output_shape),
                labels.view(-1)
            )
        
        return Seq2SeqLMOutput(logits=logits, loss=loss)


config = PretrainedConfig()
model = EncoderDecoderModel(config)

split_dataset = ds.train_test_split(test_size=0.3, seed=42) # Seed for reproducibility

# Access the resulting splits
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# Train
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds,
    compute_metrics=compute_metrics_three
)

# trainer.train()
# stats = trainer.evaluate(test_dataset)
# print(stats)
# graph(stats)

# Test Results

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics_three
)

trainer.train()
stats = trainer.evaluate(test_dataset)
print(stats)
graph(stats)






