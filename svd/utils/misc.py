import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from svd.core.mining import get_tokenizer

# from gensim.models.word2vec import Word2Vec
# from sctokenizer import CTokenizer
# from sctokenizer import TokenType


def get_vectors(w2v_keyed_vectors, tokenized_code):
    vectors = []
    no_tokens = 0

    for token in tokenized_code:
        if token[0] in w2v_keyed_vectors:
            vecs = w2v_keyed_vectors[token[0]]
            # vecs = np.array(vecs) * token[1]

            vectors.append(vecs)
        else:
            no_tokens += 1
            vectors.append(np.zeros(w2v_keyed_vectors.vector_size))
    perc = no_tokens / len(tokenized_code)

    # if perc > 0.1:
    #    print(perc)

    # The node's source embedding is the average of it's embedded tokens
    return np.mean(np.array(vectors), 0)


# Loads dataset and cleans from duplicate functions
def curate(raw_data_path, func_size: int, ratio: float = 1.0):
    tokenizer = get_tokenizer(vocab_size=None)
    print("Curating dataset")
    raw = pd.read_json(raw_data_path)
    print(f"Total: {len(raw)}, Safe: {len(raw[raw['target'] == 0])} , Unsafe: {len(raw[raw['target'] == 1])}")
    filtered = raw.drop_duplicates(subset="func", keep=False)
    # filter row by func token size
    mask = filtered.apply(lambda row: len(tokenizer(row.func)) < func_size, axis=1)
    filtered = filtered[mask]

    for key in ["commit_id", "project"]:
        del filtered[key]

    if ratio < 1:
        return filtered[:round(ratio * len(filtered))]

    print(
        f"Total: {len(filtered)}, Safe: {len(filtered[filtered['target'] == 0])} , Unsafe: {len(filtered[filtered['target'] == 1])}")

    return filtered


def hist_func_size(tokens_dataset):
    size_column = []

    for i, row in tqdm(tokens_dataset.iterrows()):
        size_column.append(len(row.tokens))

    hist, bins = np.histogram(size_column, bins=100)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.hist(size_column, bins=logbins)
    plt.title("Histogram of the size of functions in tokens")
    plt.ylabel("Frequency")
    plt.xlabel("#Tokens/Function")
    plt.xscale('log')
    plt.show()


def box_plot(tokens_dataset):
    tokens_dataset['sizes'] = tokens_dataset.tokens.apply(lambda x: len(x))
    std = tokens_dataset.sizes.std()
    threshold = 3 * std
    filtered = tokens_dataset[tokens_dataset['sizes'] < 500]
    filtered.boxplot(column=['sizes'])
    print(len(filtered), len(tokens_dataset))
    plt.show()

"""
def generate_w2v(tokens_dataset, w2v_path):
    w2vmodel = Word2Vec(size=100, alpha=0.01, workers=4, sg=1)

    # word2vec used to learn the initial embedding of each token
    print("Building w2v model")
    tokens = [[t[0] for t in ft] for ft in tokens_dataset.tokens]
    w2vmodel.build_vocab(tokens, update=False)
    print("Training w2v model")
    w2vmodel.train(tokens, total_examples=w2vmodel.corpus_count, epochs=10)
    print("Saving w2v model")
    w2vmodel.save(str(w2v_path))

    return w2vmodel
"""

"""
def get_w2v(w2v_path, tokens_dataset):
    if w2v_path.exists():
        return Word2Vec.load(str(w2v_path))
    else:
        return generate_w2v(tokens_dataset, w2v_path)
"""


def train_test_split_balanced(data_frame: pd.DataFrame, shuffle=True):
    """
      Splitting Dataset
    """

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)

    train = train_false.append(train_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = train_false.append(train_true)
    val = val_false.append(val_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test, val


# def tokenize(data_frame: pd.DataFrame):
#    data_frame.func = data_frame.func.apply(tokenizer)
# Change column name
#    data_frame = data_frame.rename(columns={'func': 'tokens'})
#    # Keep just the tokens
#    return data_frame

"""
def get_tokens(data_path, tokens_path):
    if tokens_path.exists():
        return pd.read_pickle(tokens_path)
    else:
        print("Tokenizing dataset")
        dataset = curate(data_path)
        tokenizer = CTokenizer()
        dataset['tokens'] = dataset.func.apply(
            lambda x: [(t.token_value, t.token_type.value) for t in tokenizer.tokenize(x)])
        # for i, row in tqdm(dataset.iterrows()):
        #  tokens_column.append([t.token_value for t in tokenizer.tokenize(row.func)])
        # dataset['func'] = tokens_column 
        # dataset = dataset.rename(columns={'func': 'tokens'})
        dataset.to_pickle(tokens_path)
        return dataset
"""
