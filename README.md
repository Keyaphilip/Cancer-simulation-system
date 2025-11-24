# Cancer Pathway Simulation System

This project simulates biological pathways involved in cancer, specifically focusing on the DNA Damage Response (DDR) mechanism. It allows for the simulation of genetic mutations (like BRCA1 loss-of-function) and analyzes their impact on the pathway.

## Project Structure

- **`main.py`**: The main entry point. It runs a simulation of a patient cohort, generates synthetic data, and trains a classifier to detect mutations.
- **`visualize_pathway.py`**: Visualizes the pathway graph structure.
- **`src/`**:
    - **`pathways.py`**: Defines the graph structure (nodes and edges) of the biological pathways.
    - **`simulate.py`**: Contains the logic for signal propagation through the pathway.
- **`patterns.py`**: Utilities for data vectorization and PCA analysis.
- **`train.py`**: Contains the machine learning logic (Logistic Regression) to classify simulation states.

## How to Run

1.  **Run the Simulation and Training**:
    ```bash
    python main.py
    ```
    This will output the nodes, a sample state, PCA variance, and the classifier's accuracy.

2.  **Visualize the Pathway**:
    ```bash
    python visualize_pathway.py
    ```
    This will generate a `pathway_graph.png` image showing the interactions between genes. Green arrows indicate activation, and red lines indicate inhibition.

## Next Steps

- **Expand the Pathway**: Add more genes and interactions to `src/pathways.py`.
- **Therapeutic Screening**: Implement a module to simulate drug effects.
- **Web Interface**: Build a dashboard to interactively explore the simulation.
