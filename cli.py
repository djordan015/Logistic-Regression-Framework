import csv
import numpy as np
import pandas as pd
from python.bindings import LogisticRegression


# ── CSV helper ────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np


def load_csv(path, label_col):
    # 1. Load the data
    df = pd.read_csv(path)
    
    # Clean column names (removes accidental spaces from headers)
    df.columns = df.columns.str.strip()
    
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available: {list(df.columns)}")

    # 2. Convert categorical label to 0/1
    # We check if it's already numeric. If not, we encode it.
    if not pd.api.types.is_numeric_dtype(df[label_col]):
        df[label_col] = df[label_col].astype('category')
        categories = df[label_col].cat.categories.tolist()
        
        if len(categories) != 2:
            raise ValueError(f"Expected 2 classes, found: {categories}")
            
        y = df[label_col].cat.codes.values.astype(float)
        print(f"  Mapping: {categories[0]} -> 0, {categories[1]} -> 1")
    else:
        # If already numeric (0/1), just grab the values
        y = df[label_col].values.astype(float)
        categories = None # Or list(df[label_col].unique())

    # 3. Extract Features
    X_df = df.drop(columns=[label_col])
    feature_names = X_df.columns.tolist()
    
    # Convert features to a NumPy float array
    X = X_df.values.astype(float)

    return X, y, feature_names
    

# ── Prompt helpers ────────────────────────────────────────────────────────────

def prompt(message, default=None):
    """Print a prompt and return the user's input, falling back to default."""
    suffix = f' [{default}]' if default is not None else ''
    val = input(f'{message}{suffix}: ').strip()
    return val if val else (str(default) if default is not None else '')


def prompt_file(message):
    """Keep asking until the user provides a path to an existing file."""
    while True:
        path = input(f'{message}: ').strip()
        try:
            open(path).close()
            return path
        except FileNotFoundError:
            print(f"  File not found: '{path}'. Please try again.")


def save_module():
    # ── Save weights ──────────────────────────────────────────────────────────
    print('-- Save weights --')
    save_path = prompt('Save weights to .json (press Enter to skip)', default='')

    if save_path:
        model.save_weights(save_path)
        print(f'  Weights saved to {save_path}')
        
        
def train():
    # ── Training ──────────────────────────────────────────────────────────────
    print('-- Training --')
    train_path = prompt_file('Training CSV path')
    label_col  = prompt('Label column name')  
      
    try:
        X_train, y_train, feature_names = load_csv(train_path, label_col)
    except ValueError as e:
        print(f'Error: {e}')
        return

    print(f'  Loaded {len(X_train)} samples, {len(feature_names)} features.')

    epochs    = int(prompt('Epochs',               default=1000))
    optimizer = prompt('Optimizer (gd / sgd)',     default='gd')
    debug     = prompt('Debug logging (y/n)',      default='n').lower() == 'y'
    th        = float(prompt('Threshold',                 default=0.5))

    print('\nTraining...')
    model = LogisticRegression(threshold=th, epochs=epochs, optimizer=optimizer, debug=debug)
    model.fit(X_train, y_train)
    print(f'Training accuracy: {model.score(X_train, y_train):.2%}\n')
    
    
def test():
    print('-- Test on unseen data --')
    test_path = prompt('Test CSV path (press Enter to skip)', default='')

    if test_path:
        try:
            X_test, y_test, _ = load_csv(test_path, label_col)
            print(f'Test accuracy: {model.score(X_test, y_test):.2%}\n')
        except (FileNotFoundError, ValueError) as e:
            print(f'  Error: {e}\n')

def menu():  
    print("What would you like to do?")
    valid = False
    while not valid:
        print("(1) Train new model")
        print("(2) Load existing model")
        print("(q) quit")
        mode = prompt("mode")
        
        if mode.isdigit():
            if int(mode) == 1:
                print()
                valid = True
                train()
                save_module()
                
            elif int(mode) == 2:
                print()
                print("-- Load existing Model --")
                valid = True
                snapshot_path = prompt("Path to weights: ")
                
                model = LogisticRegression()
                model.load_weights(snapshot_path)
                print(model.bias)
                print(model.weights)
                
                print("If the data is labeled, please enter the label name. Otherwise, press enter to continue")
                
                test()
                
            else:
                print("Inavalid. please enter one of the following choices.\n")
                
        elif mode.lower() == 'q':
            print()
            valid = True
            
        else:
            print("Inavalid. please enter one of the following choices.\n")
            

    
# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    print('=== Logistic Regression CLI ===\n')
    menu()
    print("Thank you for using 637-Logistic Regression Framework")

if __name__ == '__main__':
    main()
