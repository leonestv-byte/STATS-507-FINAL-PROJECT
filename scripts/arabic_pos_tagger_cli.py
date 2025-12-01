from datasets import load_dataset
from datasets import Dataset
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM
import re
from transformers import PreTrainedModel
from transformers import PretrainedConfig
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch.utils.data import TensorDataset
from transformers.modeling_outputs import TokenClassifierOutput
import torch


#####################################
######	ID To Label Mapping	#########
#####################################

id2label = []

##################################
#########	Tokenizer    #########
##################################

model_name = "UBC-NLP/AraT5v2-base-1024"
tokenizer = AutoTokenizer.from_pretrained(model_name)
max_length = 128

##################################
#########	Dataset      #########
##################################

egy_ds = load_dataset("QCRI/arabic_pos_dialect", "egy")
ds = egy_ds

label2id = {label:i for i,label in enumerate(id2label)}


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
    
    # Pad
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


# Preprocess the ds
rows = [preprocess_example(ex) for ex in ds["train"]]
# conver the ds preprocess on the tokenizer to a df
df = pd.DataFrame(rows)
# conver the tokenization df to a new ds
ds = Dataset.from_pandas(df.copy())

# Vocab Size
vocab_size = 44800

id2label = list(label2id.keys())
num_pos_tags = len(label2id)


#####################################
#########	Pad Dataset     #########
#####################################

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
        mask += [0] * (max_length- len(mask))
        attention_mask.append(mask)

    # Convert to tensors
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }

ds = ds.map(fix_dataset, batched=True)
ds.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])



###############################################
#########	BiLSTM Neural Network     #########
###############################################

output_shape = num_pos_tags

class BiLSTMCustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(vocab_size, 206)
        self.bilstm = nn.LSTM(206, 200, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(400, num_pos_tags)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embed(input_ids)
        lstm_out, _ = self.bilstm(x)
        logits = self.linear(lstm_out)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, num_pos_tags), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits
            )



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


config = PretrainedConfig()
model = BiLSTMCustomModel(config)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    eval_dataset=ds
)


trainer.train()

#################################################
######	Run a Loop to Tag Arabic Words	#########
#################################################


resume = True
terminate_signal = "END"

while resume:
	arabic_utterance = input("please print an arabic utterance, or END to end: ")
	if (arabic_utterance == terminate_signal.upper() or arabic_utterance == terminate_signal.lower()):
		print("Have a nice day!")
		break
	words = re.findall(r'\w+|[^\w\s]', arabic_utterance, re.UNICODE)
	inputs = tokenizer(
		words,
		max_length=max_length,
		truncation=True,
		is_split_into_words=True
		)
	padding_length = max_length - len(inputs['input_ids'])
	# # Pad input_ids and attention_mask
	inputs['input_ids'] = inputs['input_ids'] + [0] * padding_length
	inputs['attention_mask'] = inputs['attention_mask'] + [0] * padding_length
	# Predict
	predict_dataset = Dataset.from_dict({
		'input_ids': inputs['input_ids'],
		'attention_mask': inputs['attention_mask']
		})
	predictions = trainer.predict(predict_dataset)
	logits = predictions.predictions
	print(logits.shape)
	pred_ids = np.argmax(logits, axis=-1) 
	pos_labels = [id2label[i] for i in pred_ids]
	masked_labels = [
	label for label, m in zip(pos_labels, inputs['attention_mask']) if m == 1
	]
	output = str(words) + "\n -------> \n " + str(masked_labels)
	print(output)









