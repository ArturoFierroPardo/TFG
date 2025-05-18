
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

def modelo3_twi_owo(text, epochs, n_words, seed_text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    encoded = tokenizer.texts_to_sequences([text])[0]

    sequences = [(encoded[i-2:i], encoded[i]) for i in range(2, len(encoded))]
    X = np.array([s[0] for s in sequences])
    y = to_categorical([s[1] for s in sequences], num_classes=len(tokenizer.word_index)+1)

    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10))
    model.add(LSTM(50))
    model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, verbose=0)

    def generate_seq(model, tokenizer, seed_text, n_words):
        result = seed_text.split()
        in_text = seed_text
        for _ in range(n_words):
            encoded = tokenizer.texts_to_sequences([in_text.split()[-2:]])[0]
            encoded = np.array(encoded).reshape(1, 2)
            yhat = model.predict(encoded, verbose=0)
            yhat = np.argmax(yhat)
            out_word = ''
            for word, index in tokenizer.word_index.items():
                if index == yhat:
                    out_word = word
                    break
            in_text += ' ' + out_word
            result.append(out_word)
        return ' '.join(result)

    return generate_seq(model, tokenizer, seed_text, n_words)