from extract import *

# Hyperparameters
text_as_list = []
frequencies = {}
uncommon_words = set()
MIN_FREQUENCY = 7
MIN_SEQ = 5
BATCH_SIZE = 32

def extract_text(text):
    global text_as_list
    text_as_list += [w for w in text.split(' ') if w.strip() != '' or w == '\n']


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
X_train, X_test, y_train, y_test = train_test_split(valid_seqs, end_seq_words, test_size=0.05, random_state=42)
print(X_train[2:5])


