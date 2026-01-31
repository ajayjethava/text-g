"""
LSTM Text Generation - Complete Project
Run: python text_generator.py
"""

import numpy as np
import tensorflow as tf
import string
import requests
import os
import pickle

SEED_LENGTH = 100
BATCH_SIZE = 128
EPOCHS = 30
TEXT_LENGTH = 200000  

\
def load_dataset():
    print(" Loading Shakespeare dataset...")
    
   
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url, timeout=10)
        text = response.text
    except:
        
        print("⚠ Internet not available, using sample text")
        text = """
        to be or not to be that is the question whether tis nobler 
        in the mind to suffer the slings and arrows of outrageous 
        fortune or to take arms against a sea of troubles and by 
        opposing end them to die to sleep no more and by a sleep 
        to say we end the heartache and the thousand natural shocks 
        that flesh is heir to tis a consummation devoutly to be 
        wished to die to sleep to sleep perchance to dream ay 
        theres the rub for in that sleep of death what dreams may come
        """
    
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text[:TEXT_LENGTH]  
    
    print(f" Dataset loaded: {len(text)} characters")
    print(f"Sample: {text[:100]}...")
    return text

def preprocess_data(text, seq_length=100):
    
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    print(f" Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(chars)}")
    
    step = 3
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, step):
        sequences.append(text[i:i + seq_length])
        next_chars.append(text[i + seq_length])
    
    print(f" Created {len(sequences)} training sequences")
    
    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=np.bool_)
    y = np.zeros((len(sequences), vocab_size), dtype=np.bool_)
    
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            X[i, t, char_to_int[char]] = 1
        y[i, char_to_int[next_chars[i]]] = 1
    
    return X, y, vocab_size, char_to_int, int_to_char

def build_model(vocab_size, seq_length):
    
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(seq_length, vocab_size)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, X, y):
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True
        )
    ]
    
    # Train
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def generate_text(model, seed_text, char_to_int, int_to_char, seq_length, num_chars=200, temperature=0.7):
    """Generate text from seed"""
    generated = seed_text
    
    for _ in range(num_chars):
        x_pred = np.zeros((1, seq_length, len(char_to_int)))
        
        seed_slice = seed_text[-seq_length:] if len(seed_text) >= seq_length else seed_text
        for t, char in enumerate(seed_slice):
            if char in char_to_int:
                x_pred[0, t, char_to_int[char]] = 1
        
        preds = model.predict(x_pred, verbose=0)[0]
        
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds + 1e-7) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        
        next_idx = np.random.choice(len(preds), p=preds)
        next_char = int_to_char[next_idx]
        
        generated += next_char
        seed_text = seed_text[-seq_length:] + next_char
    
    return generated

def main():
    
    text = load_dataset()
    
    X, y, vocab_size, char_to_int, int_to_char = preprocess_data(text, SEED_LENGTH)
    
    model = build_model(vocab_size, SEED_LENGTH)
    
    history, model = train_model(model, X, y)
    
    print("\n" + "="*60)
    print("="*60)
    
    if os.path.exists('best_model.keras'):
        model = tf.keras.models.load_model('best_model.keras')
    
    seed_texts = [
        "to be or not to be",
        "romeo wherefore art thou",
        "all the world is a stage",
        "shall i compare thee",
        "now is the winter"
    ]
    
    for i, seed in enumerate(seed_texts):
        seed = seed.lower()[:SEED_LENGTH]
        print(f"\n Seed {i+1}: '{seed}'")
        print("-" * 40)
        
        generated = generate_text(
            model, seed, char_to_int, int_to_char, 
            SEED_LENGTH, num_chars=150, temperature=0.7
        )
        
        print(f"Generated:\n{generated}\n")
    
    model.save('text_generator.keras')
    
    with open('char_mappings.pkl', 'wb') as f:
        pickle.dump({'char_to_int': char_to_int, 'int_to_char': int_to_char}, f)
    
    print(" Done! Files saved:")
    print("  - text_generator.keras (model)")
    print("  - char_mappings.pkl (tokenizers)")
    print("  - best_model.keras (best training checkpoint)")
    
    if history:
        print(f"Final Loss: {history.history['loss'][-1]:.4f}")
        print(f"Final Accuracy: {history.history['accuracy'][-1]:.4f}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f" Error: {e}")
    except Exception as e:
        print(f" An error occurred: {e}")