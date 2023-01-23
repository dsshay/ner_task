import argparse
import os
import logging
import wandb
import wget
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.cuda.amp
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import seqeval.metrics

from preprocess import read_data, LABEL_TO_ID
from model import ModelConstruct

ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] + [-1] * (max_len - len(f["labels"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    output = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }
    return output


def train(args, model, train_features, test_features):
    train_dataloader = DataLoader(train_features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    logging.info("Total steps %s" % total_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=args.eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        logging.info(f"Start epoch #{epoch}")
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            if num_steps < args.alpha_warmup_ratio * total_steps:
                args.alpha_t = 0.0
            else:
                args.alpha_t = args.alpha
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
            loss = outputs[0] / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if step % args.gradient_accumulation_steps == 0:
                num_steps += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                logging.info("\nStep {}, loss {}".format(num_steps, loss.item()))
                wandb.log({'loss': loss.item()}, step=num_steps)
            if step == len(train_dataloader) - 1:
                results, fig = evaluate(args, model, test_features, tag="test")
                logging.info("\nEvaluate result epoch {}: {}".format(epoch, results))
                wandb.log(results, step=num_steps)
                wandb.log({"chart": fig})
        logging.info(f"End epoch #{epoch}")


def evaluate(args, model, features, tag="dev"):
    dataloader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
    preds, keys = [], []
    confusion = torch.zeros(len(LABEL_TO_ID), len(LABEL_TO_ID))
    for batch in dataloader:
        model.eval()
        batch = {key: value.to(args.device) for key, value in batch.items()}
        keys += batch['labels'].cpu().numpy().flatten().tolist()
        batch['labels'] = None
        with torch.no_grad():
            logits = model(**batch)[0]
            preds += np.argmax(logits.cpu().numpy(), axis=-1).tolist()

    preds, keys = list(zip(*[[pred, key] for pred, key in zip(preds, keys) if key != -1]))
    preds = [ID_TO_LABEL[pred] for pred in preds]
    keys = [ID_TO_LABEL[key] for key in keys]
    model.zero_grad()

    f1 = seqeval.metrics.f1_score([keys], [preds])
    recall = seqeval.metrics.recall_score([keys], [preds])
    precision = seqeval.metrics.precision_score([keys], [preds])

    output = {
        tag + "_f1": f1,
        tag + "_precision": precision,
        tag + "_recall": recall,
    }

    for true, pred in zip(keys, preds):
        confusion[LABEL_TO_ID[true]][LABEL_TO_ID[pred]] += 1

    for i in range(len(ID_TO_LABEL)):
        confusion[i] = confusion[i] / confusion[i].sum()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confusion.numpy())
    labels = list(LABEL_TO_ID.keys())
    ids = np.arange(len(labels))
    ax.set_ylabel('True Labels', fontsize='x-large')
    ax.set_xlabel('Pred Labels', fontsize='x-large')
    ax.set_xticks(ids)
    ax.set_xticklabels(labels)
    ax.set_yticks(ids)
    ax.set_yticklabels(labels)
    fig.tight_layout()

    return output, fig


if __name__ == "__main__":

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    wandb.init(project='NERConll2003', entity="mipt1")

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-6, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--num_class", default=9, type=int)
    parser.add_argument("--alpha", default=50.0, type=float)
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
    parser.add_argument("--model_type", default='bert-base-cased', type=str)

    args = parser.parse_args()

    test_path = './data/eng.testa'
    if not os.path.exists(test_path):
        wget.download('https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa',
                      out='./data/eng.testa')

    train_path = './data/eng.train'
    if not os.path.exists(train_path):
        wget.download('https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train',
                      out='./data/eng.train')

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    logging.info("Read and tokenize train data")
    train_features = read_data(train_path, tokenizer)
    logging.info("done")
    logging.info("Read and tokenize test data")
    test_features = read_data(test_path, tokenizer)
    logging.info("done")

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    set_seed(args.seed)

    logging.info("Initializing the model")
    model = ModelConstruct(args)
    logging.info("done")

    train(args, model, train_features, test_features)
