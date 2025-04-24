from models import EmailClassifier
import argparse

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

if __name__ == "__main__":
    main()
