
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np

def modelo1_owi_owo(text, epochs, n_words, seed_text):
    # Tokenización
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    encoded = tokenizer.texts_to_sequences([text])[0]

    # Crear x, y
    x = np.array(encoded[:-1]).reshape(-1, 1)
    y = to_categorical(encoded[1:], num_classes=len(tokenizer.word_index)+1)

    # Definir modelo
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=10))
    model.add(LSTM(50))
    model.add(Dense(len(tokenizer.word_index)+1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x, y, epochs=epochs, verbose=0)

    # Generar texto dentro de la misma función
    result = seed_text.split()
    in_text = seed_text
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0][-1:]
        encoded = np.array(encoded).reshape(1, 1)
        yhat = model.predict(encoded, verbose=0)
        yhat_index = np.argmax(yhat)
        for word, index in tokenizer.word_index.items():
            if index == yhat_index:
                result.append(word)
                in_text = word
                break

    return ' '.join(result)