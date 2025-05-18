
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

def modelo2_lbl(text, epochs, n_words, seed_text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    encoded = tokenizer.texts_to_sequences([text])[0]

    sequences = [encoded[:i] for i in range(2, len(encoded)+1)]
    max_len = max(len(s) for s in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    X, y = sequences[:, :-1], to_categorical(sequences[:, -1], num_classes=len(tokenizer.word_index)+1)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10))
    model.add(LSTM(50))
    model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=0)

    def generate_seq(model, tokenizer, seed_text, n_words, max_length):
        result = seed_text.split()
        in_text = seed_text
        for _ in range(n_words):
            encoded = tokenizer.texts_to_sequences([in_text])[0]
            encoded = pad_sequences([encoded], maxlen=max_length-1, padding='pre')
            yhat = model.predict(encoded, verbose=0)
            yhat_index = np.argmax(yhat)
            for word, index in tokenizer.word_index.items():
                if index == yhat_index:
                    result.append(word)
                    in_text += ' ' + word
                    break
        return ' '.join(result)

    return generate_seq(model, tokenizer, seed_text, n_words, max_len)