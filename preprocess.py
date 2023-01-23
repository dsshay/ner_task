from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import wandb

LABEL_TO_ID = {'O': 0, 'B-MISC': 1, 'I-MISC': 2, 'B-PER': 3, 'I-PER': 4, 'B-ORG': 5, 'I-ORG': 6, 'B-LOC': 7, 'I-LOC': 8}


def visualize_distr(data: Counter):
    labels = list(data.keys())
    plt.bar(x=labels, height=list(data.values()))
    wandb.log({"chart": wandb.Image(plt)})


def process_instance(words, labels, tokenizer, max_seq_length=512):
    tokens, token_labels = [], []
    for word, label in zip(words, labels):
        tokenized = tokenizer.tokenize(word)
        token_label = [LABEL_TO_ID[label]] + [-1] * (len(tokenized) - 1)
        tokens += tokenized
        token_labels += token_label
    assert len(tokens) == len(token_labels)
    tokens, token_labels = tokens[:max_seq_length - 2], token_labels[:max_seq_length - 2]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
    token_labels = [-1] + token_labels + [-1]
    return {
        "input_ids": input_ids,
        "labels": token_labels
    }


def read_data(file, tokenizer, max_seq_length=512):
    sentence, labels = [], []
    examples = []
    labels_fig = []
    lines = open(file, "r").readlines()
    for line in tqdm(lines):
        line = line.strip()
        if line.startswith("-DOCSTART-"):
            continue
        if len(line) > 0:
            line = line.split()
            word = line[0]
            label = line[-1]
            # For IO tagging
            # if label != 'O':
            #     label = label.split('-')[-1]
            sentence.append(word)
            labels.append(label)
            labels_fig.append(label)
        else:
            if len(sentence) > 0:
                assert len(sentence) == len(labels)
                examples.append(process_instance(sentence, labels, tokenizer, max_seq_length))
                sentence, labels = [], []
    visualize_distr(Counter(labels_fig))
    return examples
