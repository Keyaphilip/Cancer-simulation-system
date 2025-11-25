import numpy as np
from src.pathways import make_example_ddr
from main import simulate_cohort
from patterns import vectorize
from train import train_classifier
from sklearn.metrics import accuracy_score, classification_report

def test_model():
    print("1. Generating Training Data...")
    G = make_example_ddr()
    # Train on 100 samples
    nodes, states_train, y_train = simulate_cohort(G, n_samples=100, brca_lof_rate=0.5, noise=0.1)
    X_train = vectorize(states_train, nodes)
    
    print("2. Training Model...")
    clf, mean_acc, std_acc = train_classifier(X_train, y_train)
    print(f"   Training CV Accuracy: {mean_acc:.2f} +/- {std_acc:.2f}")

    print("\n3. Generating Testing Data (New Cohort)...")
    # Generate a completely new set of 50 samples
    _, states_test, y_test = simulate_cohort(G, n_samples=50, brca_lof_rate=0.5, noise=0.1)
    X_test = vectorize(states_test, nodes)

    print("4. Testing Model...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"   Test Set Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Wild Type", "BRCA1 Mutated"]))

if __name__ == "__main__":
    test_model()
