from models import EmailClassifier
import argparse
import pandas as pd
def main():
    parser = argparse.ArgumentParser(description="Train email classification model")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text_col", type=str, default="email", help="Name of text column")
    parser.add_argument("--label_col", type=str, default="type", help="Name of label column")
    
    args = parser.parse_args()
    
    # Initialize and train classifier
    classifier = EmailClassifier()
    classifier.train_from_csv(
        csv_path=args.csv_path,
        text_col=args.text_col,
        label_col=args.label_col
    )

from nlpaug import Augmenter
from nlpaug.util import Action

def augment_data(df, text_col, label_col, n_aug=3):
    aug = Augmenter(
        action=Action.SUBSTITUTE,
        name='word2vec',
        aug_min=1,
        aug_max=3
    )
    
    augmented = []
    for _, row in df.iterrows():
        for _ in range(n_aug):
            new_text = aug.augment(row[text_col])
            augmented.append({text_col: new_text, label_col: row[label_col]})
    
    return pd.concat([df, pd.DataFrame(augmented)], ignore_index=True)

if __name__ == "__main__":
    main()