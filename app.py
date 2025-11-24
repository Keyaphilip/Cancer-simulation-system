import streamlit as st
import numpy as np
import pandas as pd
from src.pathways import make_example_ddr
from src.simulate import apply_mutations, propagate
from visualize_pathway import draw_pathway
from patterns import vectorize
from train import train_classifier
from main import simulate_cohort

st.set_page_config(page_title="Cancer Pathway Simulation", layout="wide")

st.title("🧬 Cancer Pathway Simulation System")
st.markdown("""
This dashboard simulates the **DNA Damage Response (DDR)** pathway. 
You can introduce mutations (e.g., Knockout BRCA1) and observe how the signals propagate through the network.
""")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Parameters")

# Mutation Controls
st.sidebar.subheader("Mutations")
brca_status = st.sidebar.radio("BRCA1 Status", ["Wild Type", "Loss of Function (Knockout)"])
p53_status = st.sidebar.radio("p53 Status", ["Wild Type", "Loss of Function (Knockout)"])

# Simulation Noise
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.1)

# --- Simulation Logic ---
G = make_example_ddr()

# Define mutations based on UI
mutations = {"DNA_damage": 1.0} # Always trigger DNA damage for this scenario
if brca_status == "Loss of Function (Knockout)":
    mutations["BRCA1"] = -1.0
if p53_status == "Loss of Function (Knockout)":
    mutations["p53"] = -1.0

# Run single simulation for visualization
priors = apply_mutations(G, mutations)
state = propagate(G, priors)
# Add noise
state_noisy = {k: v + np.random.normal(0, noise_level) for k, v in state.items()}

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Pathway Visualization")
    # Draw graph with colors based on state
    fig = draw_pathway(G, values=state_noisy)
    st.pyplot(fig)

with col2:
    st.subheader("Node Activity Levels")
    # Display state as a dataframe/table
    df_state = pd.DataFrame(list(state_noisy.items()), columns=["Node", "Activity"])
    df_state = df_state.sort_values(by="Activity", ascending=False)
    
    # Color formatting for the table
    def color_activity(val):
        color = 'red' if val > 0.5 else 'blue' if val < -0.5 else 'black'
        return f'color: {color}'
    
    st.dataframe(df_state.style.applymap(color_activity, subset=['Activity']), use_container_width=True)

    st.subheader("Prediction")
    # Train a quick classifier on the fly (or load a pre-trained one)
    # For demo purposes, we'll retrain on a small cohort to show the concept
    with st.spinner("Training classifier on synthetic cohort..."):
        nodes_list, states_train, y_train = simulate_cohort(G, n_samples=200, brca_lof_rate=0.5, noise=noise_level)
        X_train = vectorize(states_train, nodes_list)
        clf, _, _ = train_classifier(X_train, y_train)
    
    # Predict current state
    # Ensure order matches
    current_vector = np.array([[state_noisy.get(n, 0.0) for n in nodes_list]])
    prob = clf.predict_proba(current_vector)[0]
    
    st.write("Probability of BRCA1 LOF:")
    st.progress(prob[1])
    st.write(f"**{prob[1]*100:.1f}%**")
    
    if prob[1] > 0.5:
        st.error("Prediction: BRCA1 Mutation Detected")
    else:
        st.success("Prediction: Wild Type")

st.markdown("---")
st.markdown("*Blue nodes indicate inhibition/low activity. Red nodes indicate activation/high activity.*")
