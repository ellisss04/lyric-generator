from model import *

from tensorflow import keras
from keras.models import Sequential
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding

# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size):
    index = 0
    while True:
        x = np.zeros((batch_size, MIN_SEQ), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)
    # Randomly pick a seed sequence
    seed_index = np.random.randint(len(X_train + X_test))
    seed = (X_train + X_test)[seed_index]

    for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
        sentence = seed
        examples_file.write('----- Diversity:' + str(diversity) + '\n')
        examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
        examples_file.write(' '.join(sentence))
        for i in range(50):
            x_pred = np.zeros((1, MIN_SEQ))
            for t, word in enumerate(sentence):
                x_pred[0, t] = word_indices[word]
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            sentence = sentence[1:]
            sentence.append(next_word)

            examples_file.write(" " + next_word)
        examples_file.write('\n')
    examples_file.write('=' * 80 + '\n')
    examples_file.flush()