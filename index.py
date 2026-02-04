"""
Enhanced LSTM Text Generator with Paragraphs
Run: python text_generator_enhanced.py
"""

import numpy as np
import tensorflow as tf
import random
import pickle
import os

def load_large_sample_text():
    """Larger sample text with paragraphs"""
    text = """
    to be or not to be that is the question whether tis nobler 
    in the mind to suffer the slings and arrows of outrageous 
    fortune or to take arms against a sea of troubles and by 
    opposing end them to die to sleep no more and by a sleep 
    to say we end the heartache and the thousand natural shocks 
    that flesh is heir to tis a consummation devoutly to be 
    wished to die to sleep to sleep perchance to dream ay 
    theres the rub for in that sleep of death what dreams may come
    
    romeo romeo wherefore art thou romeo deny thy father and refuse thy name
    or if thou wilt not be but sworn my love and ill no longer be a capulet
    shall i compare thee to a summers day thou art more lovely and more temperate
    rough winds do shake the darling buds of may and summers lease hath all too short a date
    
    all the worlds a stage and all the men and women merely players
    they have their exits and their entrances and one man in his time plays many parts
    
    friends romans countrymen lend me your ears i come to bury caesar not to praise him
    the evil that men do lives after them the good is oft interred with their bones
    
    now is the winter of our discontent made glorious summer by this sun of york
    and all the clouds that lourd upon our house in the deep bosom of the ocean buried
    
    a horse a horse my kingdom for a horse but soft what light through yonder window breaks
    it is the east and juliet is the sun arise fair sun and kill the envious moon
    
    if music be the food of love play on give me excess of it that surfeiting
    the appetite may sicken and so die that strain again it had a dying fall
    
    we are such stuff as dreams are made on and our little life is rounded with a sleep
    the fault dear brutus is not in our stars but in ourselves that we are underlings
    
    what a piece of work is a man how noble in reason how infinite in faculty
    in form and moving how express and admirable in action how like an angel
    
    tomorrow and tomorrow and tomorrow creeps in this petty pace from day to day
    to the last syllable of recorded time and all our yesterdays have lighted fools
    
    the quality of mercy is not strained it droppeth as the gentle rain from heaven
    upon the place beneath it is twice blest it blesseth him that gives and him that takes
    """
    
    text = text.lower()
    text = text.replace('\n', ' ')
    text = ' '.join(text.split())  
    return text

def create_model(vocab_size, seq_length=100):
    """Create enhanced LSTM model"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(seq_length, vocab_size)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

def prepare_data_with_paragraphs(text, seq_length=100, step=3):
    """Prepare training data preserving sentence structure"""
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}
    
    vocab_size = len(chars)
    
    print(f"Vocabulary includes: {''.join(chars)}")
    print(f"Punctuation available: {[c for c in chars if c in '.,!?;:']}")
    
    # Create sequences
    sequences = []
    next_chars = []
    
    for i in range(0, len(text) - seq_length, step):
        seq = text[i:i + seq_length]
        next_char = text[i + seq_length]
        
        # Only add sequences that end with word boundaries for better learning
        if len(seq) >= 2 and seq[-1] == ' ':
            sequences.append(seq)
            next_chars.append(next_char)
    
    print(f"Created {len(sequences)} training sequences")
    
    # Convert to one-hot
    X = np.zeros((len(sequences), seq_length, vocab_size), dtype=bool)
    y = np.zeros((len(sequences), vocab_size), dtype=bool)
    
    for i, seq in enumerate(sequences):
        for t, char in enumerate(seq):
            if char in char_to_int:
                X[i, t, char_to_int[char]] = 1
        
        if next_chars[i] in char_to_int:
            y[i, char_to_int[next_chars[i]]] = 1
    
    return X, y, char_to_int, int_to_char, vocab_size

def generate_paragraphs(model, seed_text, char_to_int, int_to_char, seq_length, 
                       num_paragraphs=2, sentences_per_para=3, temperature=0.7):
    """Generate full paragraphs"""
    print(f"\nGenerating {num_paragraphs} paragraphs...")
    
    all_generated = seed_text
    
    for para_num in range(num_paragraphs):
        print(f"\nParagraph {para_num + 1}:")
        print("-" * 60)
        
        paragraph = ""
        sentences_generated = 0
        
        while sentences_generated < sentences_per_para:
            # Prepare current seed
            current_seed = all_generated[-seq_length:] if len(all_generated) >= seq_length else all_generated
            current_seed = current_seed.ljust(seq_length, ' ')  
            
            # Prepare input
            x_pred = np.zeros((1, seq_length, len(char_to_int)))
            for t, char in enumerate(current_seed):
                if char in char_to_int:
                    x_pred[0, t, char_to_int[char]] = 1
            
            # Predict
            preds = model.predict(x_pred, verbose=0)[0]
            
            # Apply temperature
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds + 1e-7) / temperature
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            # Sample next character
            next_idx = np.random.choice(len(preds), p=preds)
            next_char = int_to_char[next_idx]
            
            paragraph += next_char
            all_generated += next_char
            
            # Check for sentence end
            if next_char in '.!?':
                sentences_generated += 1
                
                # Sometimes add newline after sentence
                if random.random() > 0.7:
                    paragraph += ' '
        
        # Format and print paragraph
        para_lines = format_paragraph(paragraph, line_width=70)
        for line in para_lines:
            print(f"  {line}")
        
        print()
    
    return all_generated

def format_paragraph(text, line_width=70):
    """Format text into nicely wrapped lines"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + (1 if current_line else 0) <= line_width:
            current_line.append(word)
            current_length += len(word) + (1 if current_line else 0)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def interactive_generation():
    """Main interactive function"""
    print("\n" + "="*60)
    print(" ENHANCED TEXT GENERATOR WITH PARAGRAPHS")
    print("="*60)
    
    # Load larger text
    print("\n Loading training text...")
    text = load_large_sample_text()
    print(f" Loaded {len(text)} characters")
    print(f"Sample: {text[:100]}...")
    
    # Prepare data
    seq_length = 100
    print(f"\n Preparing data with sequence length: {seq_length}")
    X, y, char_to_int, int_to_char, vocab_size = prepare_data_with_paragraphs(text, seq_length)
    
    # Create model
    print("\n Creating neural network...")
    model = create_model(vocab_size, seq_length)
    
    # Train
    print("\n Training model (this may take a few minutes)...")
    print("   Training for better paragraph generation...")
    
    model.fit(
        X, y,
        batch_size=128,
        epochs=20,
        validation_split=0.2,
        verbose=1
    )
    
    # Save mappings
    with open('char_maps.pkl', 'wb') as f:
        pickle.dump({
            'char_to_int': char_to_int,
            'int_to_char': int_to_char,
            'seq_length': seq_length
        }, f)
    
    # Interactive menu
    while True:
        print("\n" + "="*60)
        print(" GENERATION MENU")
        print("="*60)
        print("1. Generate paragraphs")
        print("2. Generate single text")
        print("3. View vocabulary")
        print("4. Test different temperatures")
        print("5. Save current model")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '6':
            print("\n Goodbye!")
            break
        
        elif choice == '3':
            print(f"\nVocabulary ({vocab_size} characters):")
            print(''.join(sorted(char_to_int.keys())))
            continue
        
        elif choice == '4':
            print("\n Temperature Testing")
            seed = input("Enter seed text: ").lower().strip() or "to be or not to be"
            
            temps = [0.3, 0.5, 0.7, 1.0, 1.5]
            for temp in temps:
                print(f"\nTemperature: {temp}")
                print("-" * 40)
                
                # Generate short sample
                generated = seed
                for _ in range(100):
                    x_pred = np.zeros((1, seq_length, vocab_size))
                    seed_slice = generated[-seq_length:] if len(generated) >= seq_length else generated
                    seed_slice = seed_slice.ljust(seq_length, ' ')
                    
                    for t, char in enumerate(seed_slice):
                        if char in char_to_int:
                            x_pred[0, t, char_to_int[char]] = 1
                    
                    preds = model.predict(x_pred, verbose=0)[0]
                    preds = np.asarray(preds).astype('float64')
                    preds = np.log(preds + 1e-7) / temp
                    exp_preds = np.exp(preds)
                    preds = exp_preds / np.sum(exp_preds)
                    
                    next_idx = np.random.choice(len(preds), p=preds)
                    generated += int_to_char[next_idx]
                
                # Print first 80 chars
                print(generated[:80] + "...")
            continue
        
        elif choice == '5':
            model.save('paragraph_generator.h5')
            print(" Model saved as 'paragraph_generator.h5'")
            continue
        
        # Get generation parameters
        print("\n Generation Parameters:")
        seed_text = input("Enter seed text (or Enter for default): ").lower().strip()
        if not seed_text:
            seed_text = "to be or not to be"
        
        temp = float(input("Temperature (0.3-1.5, default 0.7): ") or "0.7")
        
        if choice == '1':
            num_paragraphs = int(input("Number of paragraphs (1-5): ") or "2")
            sentences_per_para = int(input("Sentences per paragraph (2-8): ") or "4")
            
            print("\n" + "="*60)
            print(f"Generating {num_paragraphs} paragraphs from: '{seed_text}'")
            print("="*60)
            
            generated = generate_paragraphs(
                model, seed_text, char_to_int, int_to_char, seq_length,
                num_paragraphs, sentences_per_para, temp
            )
            
        elif choice == '2':
            length = int(input("Number of characters to generate (100-1000): ") or "300")
            
            print("\n" + "="*60)
            print(f"Generating {length} characters from: '{seed_text}'")
            print("="*60)
            
            generated = seed_text
            for i in range(length):
                x_pred = np.zeros((1, seq_length, vocab_size))
                seed_slice = generated[-seq_length:] if len(generated) >= seq_length else generated
                seed_slice = seed_slice.ljust(seq_length, ' ')
                
                for t, char in enumerate(seed_slice):
                    if char in char_to_int:
                        x_pred[0, t, char_to_int[char]] = 1
                
                preds = model.predict(x_pred, verbose=0)[0]
                preds = np.asarray(preds).astype('float64')
                preds = np.log(preds + 1e-7) / temp
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                
                next_idx = np.random.choice(len(preds), p=preds)
                next_char = int_to_char[next_idx]
                generated += next_char
            
            # Format output
            print("\nGenerated Text:")
            print("-" * 60)
            lines = format_paragraph(generated, line_width=70)
            for line in lines:
                print(f"  {line}")
            print("-" * 60)
        
        # Save option
        save = input("\n Save generated text? (y/n): ").lower()
        if save == 'y':
            filename = input("Filename (or Enter for auto): ").strip()
            if not filename:
                filename = f"generated_{random.randint(1000, 9999)}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Seed: {seed_text}\n")
                f.write(f"Temperature: {temp}\n")
                f.write(f"Length: {len(generated)}\n\n")
                f.write(generated)
            
            print(f" Text saved to '{filename}'")

def quick_mode():
    """Quick generation without training"""
    print("\n QUICK GENERATION MODE")
    print("="*60)
    
    if os.path.exists('char_maps.pkl'):
        with open('char_maps.pkl', 'rb') as f:
            data = pickle.load(f)
        
        char_to_int = data['char_to_int']
        int_to_char = data['int_to_char']
        seq_length = data.get('seq_length', 100)
        
        vocab_size = len(char_to_int)
        model = create_model(vocab_size, seq_length)
        
        print("\nAvailable models:")
        models = [f for f in os.listdir('.') if f.endswith('.h5')]
        if models:
            for i, m in enumerate(models):
                print(f"{i+1}. {m}")
            
            model_choice = input("\nSelect model (or Enter for none): ").strip()
            if model_choice:
                model_idx = int(model_choice) - 1
                model.load_weights(models[model_idx])
                print(f" Loaded {models[model_idx]}")
        
        print("\n Quick Generation:")
        seed = input("Seed text: ").lower().strip() or "to be or not to be"
        temp = float(input("Temperature (0.7): ") or "0.7")
        length = int(input("Characters (300): ") or "300")
        
        generated = seed
        for i in range(length):
            x_pred = np.zeros((1, seq_length, vocab_size))
            seed_slice = generated[-seq_length:] if len(generated) >= seq_length else generated
            seed_slice = seed_slice.ljust(seq_length, ' ')
            
            for t, char in enumerate(seed_slice):
                if char in char_to_int:
                    x_pred[0, t, char_to_int[char]] = 1
            
            preds = model.predict(x_pred, verbose=0)[0]
            preds = np.asarray(preds).astype('float64')
            preds = np.log(preds + 1e-7) / temp
            exp_preds = np.exp(preds)
            preds = exp_preds / np.sum(exp_preds)
            
            next_idx = np.random.choice(len(preds), p=preds)
            generated += int_to_char[next_idx]
        
        print("\n" + "="*60)
        print("Generated Text:")
        print("="*60)
        lines = format_paragraph(generated, line_width=70)
        for line in lines:
            print(f"  {line}")
        print("="*60)
    else:
        print(" No pre-trained model found. Please run training mode first.")

def main():
    print("\n" + "="*60)
    print(" ENHANCED LSTM TEXT GENERATOR")
    print("="*60)
    print("This version generates full paragraphs with proper formatting!")
    
    print("\nOptions:")
    print("1. Full Training + Generation Mode")
    print("2. Quick Generation (requires pre-trained model)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        interactive_generation()
    elif choice == '2':
        quick_mode()
    else:
        print(" Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user")
    except Exception as e:
        print(f"\n Error: {e}")
        print("\nMake sure TensorFlow is installed:")
        print("pip install tensorflow numpy")
