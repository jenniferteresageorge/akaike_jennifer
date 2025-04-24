from models import EmailClassifier

clf = EmailClassifier()
clf.train_from_csv("data/combined_emails_with_natural_pii.csv", use_grid_search=True)
