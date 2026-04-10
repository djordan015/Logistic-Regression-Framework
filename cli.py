import csv
import numpy as np
from python.bindings import LogisticRegression


# ── CSV helper ────────────────────────────────────────────────────────────────

def load_csv(path, label_col):
    """
    Load a CSV file and split it into features and labels.

    The first row is expected to be a header. The column named `label_col`
    is used as the label; all other columns are treated as features.

    Returns (X, y, feature_names).
    """
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = [col.strip() for col in next(reader)]
        rows = [row for row in reader]

    if label_col not in header:
        raise ValueError(
            f"Column '{label_col}' not found.\n"
            f"  Available columns: {header}"
        )

    label_idx   = header.index(label_col)
    feature_idx = [i for i in range(len(header)) if i != label_idx]

    data = np.array(rows, dtype=float)
    X    = data[:, feature_idx]
    y    = data[:, label_idx]

    return X, y, [header[i] for i in feature_idx]


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


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    print('=== Logistic Regression CLI ===\n')

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

    print('\nTraining...')
    model = LogisticRegression(threshold=0.5, epochs=epochs, optimizer=optimizer, debug=debug)
    model.fit(X_train, y_train)
    print(f'Training accuracy: {model.score(X_train, y_train):.2%}\n')

    # ── Testing ───────────────────────────────────────────────────────────────
    print('-- Test on unseen data --')
    test_path = prompt('Test CSV path (press Enter to skip)', default='')

    if test_path:
        try:
            X_test, y_test, _ = load_csv(test_path, label_col)
            print(f'Test accuracy: {model.score(X_test, y_test):.2%}\n')
        except (FileNotFoundError, ValueError) as e:
            print(f'  Error: {e}\n')

    # ── Save weights ──────────────────────────────────────────────────────────
    print('-- Save weights --')
    save_path = prompt('Save weights to .json (press Enter to skip)', default='')

    if save_path:
        model.save_weights(save_path)
        print(f'  Weights saved to {save_path}')


if __name__ == '__main__':
    main()
