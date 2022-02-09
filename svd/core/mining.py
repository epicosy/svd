import numpy as np
import pickle
import re
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer

num_workers = 3


# Remove comments, annotations, import and empty lines from the content of a file
def extract_file_content(file_content):
    file_content = file_content.encode("utf-8", "replace").decode("utf-8")

    # Remove multi-line comments
    pattern = r"^\s*/\*(.*?)\*/"
    file_content = re.sub(pattern, "", file_content, flags=re.DOTALL | re.MULTILINE)

    # Remove inline comments within /* and */
    pattern = r"/\*(.*?)\*/"
    file_content = re.sub(pattern, "", file_content)

    # Remove single-line comments
    pattern = r"^\s*//.*"
    file_content = re.sub(pattern, "", file_content, flags=re.MULTILINE)

    # Remove empty lines
    pattern = r"^\s*[\r\n]"
    file_content = re.sub(pattern, "", file_content, flags=re.MULTILINE)

    return file_content


# Tokenization Pattern for splitting code
def gen_tok_pattern():
    single_toks = ['<=', '>=', '<', '>', '\\?', '\\/=', '\\+=', '\\-=', '\\+\\+', '--', '\\*=', '\\+', '-', '\\*',
                   '\\/', '!=', '==', '=', '!', '&=', '&', '\\%', '\\|\\|', '\\|=', '\\|', '\\$', '\\:']
    single_toks = '(?:' + '|'.join(single_toks) + ')'
    word_toks = '(?:[a-zA-Z0-9]+)'
    return single_toks + '|' + word_toks


# Build a code tokenizer or vectorizer
def get_tokenizer(vocab_size: int, vectorize=False):
    code_token_pattern = gen_tok_pattern()
    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 1), use_idf=False, max_features=vocab_size, norm=None,
                                 smooth_idf=False, lowercase=False, token_pattern=code_token_pattern, vocabulary=None)
    if vectorize:
        return vectorizer
    return vectorizer.build_analyzer()


# Train and save a BoW feature model
def train_BoW(sentences, vocab_size: int, output_dir: str, data_name: str = 'model'):
    vectorizer = get_tokenizer(vocab_size=vocab_size, vectorize=True)
    vectorizer.fit(sentences)
    pickle.dump(vectorizer, open(output_dir + "/BoW_" + data_name + ".model", "wb"))
    return vectorizer


"""
def train_w2v(sentences, dataName='model'):
    # Tokenize files
    tokenizer = get_tokenizer()
    sentences = list(map(tokenizer, sentences))
    # Train and save model
    w2v_model = Word2Vec(sentences, size=INPUT_SIZE, window=5, min_count=10, workers=num_workers, sg=1, seed=42)
    w2v_model.save("data/feature_models/w2v_" + dataName + ".model")
    return w2v_model
"""


# Convert a sentence to a feature vector using text embeddings
#  Vector averaging of words in sentence
def sen_to_vec(sen, w2v_model, input_size: int):
    sen_vec = np.array([0.0] * input_size)
    cnt = 0

    for w in sen:
        try:
            sen_vec = sen_vec + w2v_model[w]
            cnt += 1
        except:
            pass
    if cnt == 0:
        return np.array([0.0] * input_size)

    return sen_vec / (cnt * 1.0)


def infer_nlp_features(data, vocab_size: int, output_dir: str, scenario: str = 'model'):
    # Clean the data
    # data['file_contents'] = data['file_contents'].map(clean_changes)

    # Create a tokenizer
    tokenizer = get_tokenizer(vocab_size=vocab_size)

    # Load NLP feature models
    bow_model = pickle.load(open(output_dir + "/BoW_" + scenario + ".model", "rb"))

    # Remove comments and blank lines
    data["func"] = data["func"].apply(lambda x: extract_file_content(x))
    # Tokenize the features
    sentences = data['func'].apply(lambda x: np.str_(x)).map(tokenizer)

    # Encode and concatenate the features
    bow_features = bow_model.transform(data['func'].apply(lambda x: np.str_(x)).values)
    return bow_features


def save_sparse_matrix(sparse_matrix, output_dir: str, project_name: str):
    scipy.sparse.save_npz(f"{output_dir}/{project_name}_features_sparse.npz", sparse_matrix)


def mine(data, project: str, output_path: str, vocab_size: int):
    print("Cleaning...")
    # Remove comments and blank lines
    data["func"] = data["func"].astype(str)
    data["func"] = data["func"].apply(lambda x: extract_file_content(x))

    # Extract sentences.
    sentences = data["func"].apply(lambda x: np.str_(x))

    print("Training...")
    # Train the models.
    feature_model = train_BoW(sentences, vocab_size=vocab_size, output_dir=output_path, data_name=project)
    # train_w2v(sentences, dataName=project)
    # Write the Vocab
    with open(output_path+"/BoW_" + project + "_vocab.txt", 'w') as f:
        for i in feature_model.vocabulary_:
            f.write(str(i) + '\n')

    print("Inferring...")
    return infer_nlp_features(data, vocab_size=vocab_size, output_dir=output_path, scenario=project)
