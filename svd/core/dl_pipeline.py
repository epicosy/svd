from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, average_precision_score
import pickle

from svd.core.mining import gen_tok_pattern
from svd.utils.misc import train_val_test_split


def tokenize_split_dataset(train: pd.DataFrame, test: pd.DataFrame, val: pd.DataFrame, vocab_size: int, input_size: int):
    # Create source code sdata for tokenization
    x_all = train['func']

    # Tokenizer with word-level
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=gen_tok_pattern(), num_words=vocab_size, char_level=False)
    tokenizer.fit_on_texts(list(x_all))
    del (x_all)
    print('Number of tokens: ', len(tokenizer.word_counts))

    # Tokenizing train data and create matrix
    list_tokenized_train = tokenizer.texts_to_sequences(train.func)
    x_train = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_train, maxlen=input_size, padding='post')
    x_train = x_train.astype(np.int64)

    # Tokenizing test data and create matrix
    list_tokenized_test = tokenizer.texts_to_sequences(test.func)
    x_test = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_test, maxlen=input_size, padding='post')
    x_test = x_test.astype(np.int64)

    # Tokenizing validate data and create matrix
    list_tokenized_validate = tokenizer.texts_to_sequences(val['func'])
    x_val = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_validate, maxlen=input_size, padding='post')
    x_val = x_val.astype(np.int64)

    return x_train, x_test, x_val


def cnn_model(input_dim: int, input_length: int, random_weights):
    # Must use non-sequential model building to create branches in the output layer
    model = tf.keras.Sequential(name="CNN")

    model.add(tf.keras.layers.Embedding(input_dim=input_dim,
                                        output_dim=round(input_dim ** 0.25),
                                        weights=[random_weights],
                                        input_length=input_length))
    # model.add(tf.keras.layers.GaussianNoise(stddev=0.01))
    model.add(tf.keras.layers.Convolution1D(filters=128, kernel_size=(9), padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=5))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    # Define custom optimizers
    adam = tf.keras.optimizers.Adam(learning_rate=0.05, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)

    ## Compile model with metrics
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    print("CNN model built: ")
    model.summary()

    return model


def tf_callbacks(callback_dir: str, model, file_path_fmt: str):
    ## Create TensorBoard callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    tbCallback = tf.keras.callbacks.TensorBoard(log_dir=callback_dir,
                                                histogram_freq=1,
                                                embeddings_freq=1,
                                                write_graph=True,
                                                write_images=True)

    tbCallback.set_model(model)

    ## Create best model callback
    mcp = tf.keras.callbacks.ModelCheckpoint(filepath=file_path_fmt,
                                             monitor="val_loss",
                                             save_best_only=True,
                                             mode='auto',
                                             save_freq='epoch',
                                             verbose=1)

    return tbCallback, early_stop, mcp


def evaluate_model(model, x_test: pd.DataFrame, test: pd.DataFrame):
    results = model.evaluate(x_test, test.target.to_numpy(), batch_size=128)
    for num in range(0, len(model.metrics_names)):
        print(model.metrics_names[num] + ': ' + str(results[num]))

    # predicted = model.predict_classes(x_test)
    predict_x = model.predict(x_test)
    predicted = np.round(predict_x)
    predicted_prob = model.predict(x_test)

    confusion = confusion_matrix(y_true=test.target.to_numpy(), y_pred=predicted)
    print(confusion)
    tn, fp, fn, tp = confusion.ravel()
    print('\nTP:', tp)
    print('FP:', fp)
    print('TN:', tn)
    print('FN:', fn)
    report = classification_report(test.target.to_numpy(), predicted, output_dict=True)
    print(report)
    ## Performance measure
    print('\nAccuracy: ' + str(accuracy_score(y_true=test.target.to_numpy(), y_pred=predicted)))
    print('Precision: ' + str(precision_score(y_true=test.target.to_numpy(), y_pred=predicted)))
    print('Recall: ' + str(recall_score(y_true=test.target.to_numpy(), y_pred=predicted)))
    print('F-measure: ' + str(f1_score(y_true=test.target.to_numpy(), y_pred=predicted)))
    print(
        'Precision-Recall AUC: ' + str(average_precision_score(y_true=test.target.to_numpy(), y_score=predicted_prob)))
    print('AUC: ' + str(roc_auc_score(y_true=test.target.to_numpy(), y_score=predicted_prob)))
    print('MCC: ' + str(matthews_corrcoef(y_true=test.target.to_numpy(), y_pred=predicted)))


def plot_history(model, history):
    epochs_range = range(len(history.history[model.metrics_names[1]]))

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(15, 5))
    fig.suptitle('CNN with 40 Epochs, maxpool of 5, class weights of 1:5')

    axs[0].plot(epochs_range, history.history[model.metrics_names[0]], 'b', label='Loss', color='red')
    axs[0].plot(epochs_range, history.history['val_%s' % (model.metrics_names[0])], 'b', label='Val_Loss',
                color='green')

    axs[1].plot(epochs_range, history.history[model.metrics_names[1]], 'b', label='Accuracy', color='red')
    axs[1].plot(epochs_range, history.history['val_%s' % (model.metrics_names[1])], 'b', label='Val_Accuracy',
                color='green')

    axs[0].set_title('Training vs Validation loss')
    axs[0].legend()

    axs[1].set_title('Training vs Validation accuracy')
    axs[1].legend()


def train_test_cnn(dataset: pd.DataFrame, input_size: int, vocab_size: int, model_output: str, epochs: int = 40,
                   batch_size: int = 128):
    print("Tensorlfow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

    # Generate random seed
    myrand = 71926
    np.random.seed(myrand)
    tf.random.set_seed(myrand)
    print("Random seed is:", myrand)

    # dataset.target = dataset.target.apply(lambda target: False if target == 0 else True)
    train, test, val = train_val_test_split(dataset)
    x_train, x_test, x_val = tokenize_split_dataset(train, test, val, vocab_size, input_size=input_size)

    # Create a random weights matrix
    random_weights = np.random.normal(size=(vocab_size, round(vocab_size ** 0.25)), scale=0.01)
    model_output_path = Path(model_output)

    if not model_output_path.exists():
        model_output_path.mkdir(parents=True)

    callback_dir = model_output_path / 'cb'
    if not callback_dir.exists():
        callback_dir.mkdir(parents=True)

    model = cnn_model(input_dim=vocab_size, input_length=input_size, random_weights=random_weights)
    tb_callback, early_stop, mcp = tf_callbacks(model=model, callback_dir=str(callback_dir),
                                                file_path_fmt=str(
                                                    model_output_path) + "/model-epoch-100-{epoch:02d}-single.hdf5")

    history = model.fit(x=x_train,
                        y=train.target.to_numpy(),
                        validation_data=(x_val, val.target.to_numpy()),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        callbacks=[mcp, tb_callback, early_stop])

    history_path = callback_dir / "history-epochs-{epochs}-CNN-single"

    # if not history_path.exists():
    #    history_path.mkdir(parents=True)

    with history_path.open(mode='wb') as file_pi:
        pickle.dump(history.history, file_pi)

    evaluate_model(model, x_test=x_test, test=test)
    plot_history(model, history=history)

    tf.keras.backend.clear_session()
    del model
