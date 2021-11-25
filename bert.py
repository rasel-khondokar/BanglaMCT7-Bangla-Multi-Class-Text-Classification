import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import tensorflow as tf
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertPreTrainedModel, BertModel
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# get cleaned train and test data
# get cleaned train and test data
from preprocessing.preprocessing import PreProcessor
from settings import DIR_IMAGES_EDA, DIR_RESOURCES, DIR_PERFORMENCE_REPORT

MAX_LEN = 128 # max sequences length
batch_size = 32
MODEL_PRETRAINED = 'monsoon-nlp/bangla-electra'


preprocessor = PreProcessor()
df, df_test = preprocessor.read_collected_data()
num_labels = len(df.category.unique())

device_name = tf.test.gpu_device_name()

if device_name == '/device:GPU:0':
    print(f'Found GPU at: {device_name}')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU in use:', torch.cuda.get_device_name(0))
else:
    print('using the CPU')
    device = torch.device("cpu")


def preprocessing(df):
    sentences = df.cleanText.values

    # labels = np.array([labels_encoding[l] for l in df.label.values])
    with open(DIR_RESOURCES + '/label_encoder.pickle', 'rb') as handle:
        le = pickle.load(handle)
    encoded_labels = le.transform(df.category)
    labels = np.array(encoded_labels)
    class_names = le.classes_

    tokenizer = BertTokenizer.from_pretrained(MODEL_PRETRAINED, do_lower_case=True)

    encoded_sentences = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN
        )

        encoded_sentences.append(encoded_sent)
    encoded_sentences = pad_sequences(encoded_sentences, maxlen=MAX_LEN, dtype="long",
                                      value=0, truncating="post", padding="post")
    return encoded_sentences, labels


def attention_masks(encoded_sentences):
    # attention masks, 0 for padding, 1 for actual token
    attention_masks = []
    for sent in encoded_sentences:
        att_mask = [int(token_id > 0) for token_id in sent]
        attention_masks.append(att_mask)
    return attention_masks


train_encoded_sentences, train_labels = preprocessing(df)
train_attention_masks = attention_masks(train_encoded_sentences)

test_encoded_sentences, test_labels = preprocessing(df_test)
test_attention_masks = attention_masks(test_encoded_sentences)

train_inputs = torch.tensor(train_encoded_sentences)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_attention_masks)

validation_inputs = torch.tensor(test_encoded_sentences)
validation_labels = torch.tensor(test_labels)
validation_masks = torch.tensor(test_attention_masks)



# data loader for training
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# data loader for validation
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

model = BertForSequenceClassification.from_pretrained(
    MODEL_PRETRAINED,
    num_labels = num_labels,
    output_attentions = False,
    output_hidden_states = False,
)

if torch.cuda.is_available():
    model.cuda()

optimizer = AdamW(model.parameters(),
                  lr = 3e-5,
                  eps = 1e-8,
                  weight_decay = 0.01
                )

epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # 10% * datasetSize/batchSize
                                            num_training_steps = total_steps)
def compute_accuracy(preds, labels):
    p = np.argmax(preds, axis=1).flatten()
    l = labels.flatten()
    return np.sum(p==l)/len(l)


def run_train(epochs):
    losses = []
    for e in range(epochs):
        print('======== Epoch {:} / {:} ========'.format(e + 1, epochs))
        start_train_time = time.time()
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step % 10 == 0:
                elapsed = time.time() - start_train_time
                print(f'{step}/{len(train_dataloader)} --> Time elapsed {elapsed}')

            # input_data, input_masks, input_labels = batch
            input_data = batch[0].to(device)
            input_masks = batch[1].to(device)
            input_labels = batch[2].to(device)

            model.zero_grad()

            # forward propagation
            out = model(input_data,
                        token_type_ids=None,
                        attention_mask=input_masks,
                        labels=input_labels)

            loss = out[0]
            total_loss = total_loss + loss.item()

            # backward propagation
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), 1)

            optimizer.step()

        epoch_loss = total_loss / len(train_dataloader)
        losses.append(epoch_loss)
        print(f"Training took {time.time() - start_train_time}")

        # Validation
        start_validation_time = time.time()
        model.eval()
        eval_loss, eval_acc = 0, 0
        for step, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            eval_data, eval_masks, eval_labels = batch
            with torch.no_grad():
                out = model(eval_data,
                            token_type_ids=None,
                            attention_mask=eval_masks)
            logits = out[0]

            #  Uncomment for GPU execution
            logits = logits.detach().cpu().numpy()
            eval_labels = eval_labels.to('cpu').numpy()
            batch_acc = compute_accuracy(logits, eval_labels)

            # Uncomment for CPU execution
            # batch_acc = compute_accuracy(logits.numpy(), eval_labels.numpy())

            eval_acc += batch_acc
        print(f"Accuracy: {eval_acc / (step + 1)}, Time elapsed: {time.time() - start_validation_time}")
    return losses

losses = run_train(epochs)

plt.plot(losses, 'b-o')
plt.title("Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
file_path = DIR_IMAGES_EDA + '/bert_acc_loss.png'
plt.savefig(file_path)
plt.close()

output_dir = f'resources/bert_models/{MODEL_PRETRAINED.replace("/", "_")}'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_to_save = model.module if hasattr(model, 'module') else model
model_to_save.save_pretrained(output_dir)


# tokenizer.save_pretrained(output_dir)

def run_test(df_test, is_test=True):
    test_encoded_sentences, test_labels = preprocessing(df_test)
    actual_labels = test_labels
    test_attention_masks = attention_masks(test_encoded_sentences)

    test_inputs = torch.tensor(test_encoded_sentences)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_attention_masks)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    model.eval()
    eval_loss, eval_acc = 0, 0

    predicted_labels = []

    for step, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        eval_data, eval_masks, eval_labels = batch
        with torch.no_grad():
            out = model(eval_data,
                        token_type_ids=None,
                        attention_mask=eval_masks)
        logits = out[0]
        logits = logits.detach().cpu().numpy()
        eval_labels = eval_labels.to('cpu').numpy()
        predicted_labels += list(np.argmax(logits, axis=1).flatten())

        batch_acc = compute_accuracy(logits, eval_labels)
        eval_acc += batch_acc

    print(f"Accuracy: {eval_acc / (step + 1)}")
    preprocessor = PreProcessor()
    eval_labels, class_names = preprocessor.decode_category(predicted_labels)
    actual_labels, class_names = preprocessor.decode_category(list(actual_labels))
    # print(len(actual_labels), len(eval_labels))


    cm = confusion_matrix(actual_labels, eval_labels)
    report = classification_report(actual_labels, eval_labels)
    print(report)

    if is_test:
        data_split = 'test'
    else:
        data_split = 'train'

    report_filename = f'{MODEL_PRETRAINED.replace("/", "_")}'
    with open(f'{DIR_PERFORMENCE_REPORT}/{report_filename}_{data_split}.txt', 'w') as file:
        file.write('___________________ confusion_matrix _____________________\n')
        file.write(str(cm))
        file.write('\n\n\n')
        file.write('___________________ classification report _____________________\n')
        file.write(str(report))

run_test(df_test, is_test=True)
run_test(df, is_test=False)


