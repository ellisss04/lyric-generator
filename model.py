from __future__ import print_function

import numpy as np

from pre import sum_df
from sklearn.model_selection import train_test_split
from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Bidirectional, Embedding

text_as_list = []
frequencies = {}
uncommon_words = set()
MIN_FREQUENCY = 7
MIN_SEQ = 5
BATCH_SIZE = 32


def extract_text(text):
    global text_as_list
    text_as_list += [w for w in text.split(' ') if w.strip() != '' or w == '\n']


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


def get_model():
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=len(words), output_dim=1024))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


sum_df['single_text'].apply(extract_text)
print('Total words: ', len(text_as_list))
for w in text_as_list:
    frequencies[w] = frequencies.get(w, 0) + 1

uncommon_words = set([key for key in frequencies.keys() if frequencies[key] < MIN_FREQUENCY])
words = sorted(set([key for key in frequencies.keys() if frequencies[key] >= MIN_FREQUENCY]))
num_words = len(words)
word_indices = dict((w, i) for i, w in enumerate(words))
indices_word = dict((i, w) for i, w in enumerate(words))
print('Words with less than {} appearances: {}'.format(MIN_FREQUENCY, len(uncommon_words)))
print('Words with more than {} appearances: {}'.format(MIN_FREQUENCY, len(words)))
valid_seqs = []
end_seq_words = []
for i in range(len(text_as_list) - MIN_SEQ):
    end_slice = i + MIN_SEQ + 1
    if len(set(text_as_list[i:end_slice]).intersection(uncommon_words)) == 0:
        valid_seqs.append(text_as_list[i: i + MIN_SEQ])
        end_seq_words.append(text_as_list[i + MIN_SEQ])

print('Valid sequences of size {}: {}'.format(MIN_SEQ, len(valid_seqs)))
X_train, X_test, y_train, y_test = train_test_split(valid_seqs, end_seq_words, test_size=0.02, random_state=42)
print(X_train[2:5])

model = get_model()
model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
file_path = "./checkpoints/LSTM_LYRICS-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
            "loss{loss:.4f}-acc{accuracy:.4f}-val_loss{val_loss:.4f}-val_acc{val_accuracy:.4f}" % \
            (len(words), MIN_SEQ, MIN_FREQUENCY)
checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', save_best_only=True)
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=20)
callbacks_list = [checkpoint, print_callback, early_stopping]
examples_file = open('examples.txt', "w")
model.fit(generator(X_train, y_train, BATCH_SIZE),
          steps_per_epoch=int(len(valid_seqs) / BATCH_SIZE) + 1,
          epochs=20,
          callbacks=callbacks_list,
          validation_data=generator(X_test, y_train, BATCH_SIZE),
          validation_steps=int(len(y_train) / BATCH_SIZE) + 1)
