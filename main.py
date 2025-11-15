from src.pathways import make_example_ddr
from src.simulate import apply_mutations, propagate
from patterns import vectorize, embed_pca
from train import train_classifier
import numpy as np

def simulate_cohort(G, n_samples=40, brca_lof_rate=0.5, noise=0.1):
    nodes = list(G.nodes())
    states = []
    labels = []
    for i in range(n_samples):
        is_lof = (np.random.rand() < brca_lof_rate)
        muts = {"DNA_damage": 1.0}
        if is_lof:
            muts["BRCA1"] = -1.0
        priors = apply_mutations(G, muts)
        state = propagate(G, priors)
        # add small noise
        state = {k: v + np.random.normal(0, noise) for k, v in state.items()}
        states.append(state)
        labels.append(1 if is_lof else 0)
    return nodes, states, np.array(labels)

if __name__ == "__main__":
    G = make_example_ddr()
    nodes, states, y = simulate_cohort(G, n_samples=100, brca_lof_rate=0.5, noise=0.05)
    X = vectorize(states, nodes)
    Z, pca = embed_pca(X, n=2)
    clf, mean_acc, std_acc = train_classifier(X, y)
    print("Nodes:", nodes)
    print("First sample state:", states[0])
    print("PCA variance ratio:", pca.explained_variance_ratio_)
    print("Classifier CV accuracy:", mean_acc, "+/-", std_acc)
