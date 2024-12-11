import streamlit as st
import numpy as np
from scipy.stats import truncnorm

# Function to generate random values from a truncated normal distribution in the range 0-10
def truncated_normal(mean, std_dev, size, min_val=0, max_val=10):
    lower_bound = (min_val - mean) / std_dev
    upper_bound = (max_val - mean) / std_dev
    values = truncnorm.rvs(lower_bound, upper_bound, loc=mean, scale=std_dev, size=size)
    return values

# Streamlit App Title
st.title("Achievement Likelihood Simulation")

# Input Parameters
st.header("Simulation Parameters")

# User Input for Talent and Effort
st.header("Your Talent and Effort")
user_talent = st.slider("Your Talent (0-10):", min_value=0.0, max_value=10.0, value=5.0)
user_effort = st.slider("Your Effort (0-10):", min_value=0.0, max_value=10.0, value=5.0)

# Adjustable Weights Section
with st.expander("Adjust Weights for Talent, Effort, and Luck (Optional)"):
    st.write("Adjust the weights (must sum to 1).")
    weight_talent = st.slider("Weight for Talent:", min_value=0.0, max_value=1.0, value=0.24)
    weight_effort = st.slider("Weight for Effort:", min_value=0.0, max_value=1.0, value=0.24)
    weight_luck = st.slider("Weight for Luck:", min_value=0.0, max_value=1.0, value=0.52)

    # Ensure weights sum to 1
    if not np.isclose(weight_talent + weight_effort + weight_luck, 1.0):
        st.error("The weights must sum to 1. Please adjust them accordingly.")

# Target Percentile Slider
st.header("Top X% Requirement")
percentile = st.slider("Choose the Percentile to Check (1-99):", min_value=1, max_value=99, value=10)

st.header("Number of Attempts to be Made")
# Number of Attempts Sliders
attempts_1 = st.slider("Choose the First Number of Attempts (1-50):", min_value=1, max_value=50, value=10)
attempts_2 = st.slider("Choose the Second Number of Attempts (1-50):", min_value=1, max_value=50, value=20)
attempts_3 = st.slider("Choose the Third Number of Attempts (1-50):", min_value=1, max_value=50, value=30)

# Simulation Function

def simulate_user_multiple_times_chunked(talent, effort, attempts, total_simulations, chunk_size, weight_talent, weight_effort, weight_luck):
    results = []
    for _ in range(total_simulations // chunk_size):
        chunk_results = []
        for _ in range(chunk_size):
            grand_achievement = 0
            for _ in range(attempts):
                luck = truncated_normal(mean=5.0, std_dev=2, size=1, min_val=0, max_val=10)[0]
                achievement = (talent ** weight_talent) * (effort ** weight_effort) * (luck ** weight_luck)
                grand_achievement += achievement
            chunk_results.append(grand_achievement)
        results.extend(chunk_results)
    return np.array(results)

# Run Simulation
if st.button("Run Simulation"):
    num_simulations = 100000  # Fixed number of simulations
    chunk_size = 10000  # Break the simulations into chunks
    population_size = 10000  # Fixed population size

    # Generate Population Data for Comparison
    talent_population = truncated_normal(mean=5.0, std_dev=2, size=population_size, min_val=0, max_val=10)
    effort_population = truncated_normal(mean=5.0, std_dev=2, size=population_size, min_val=0, max_val=10)

    for attempts, attempt_label in zip([attempts_1, attempts_2, attempts_3], ["First", "Second", "Third"]):
        user_results = simulate_user_multiple_times_chunked(user_talent, user_effort, attempts, num_simulations, chunk_size, weight_talent, weight_effort, weight_luck)

        grand_achievements = []
        for _ in range(attempts):
            luck_population = truncated_normal(mean=5.0, std_dev=2, size=population_size, min_val=0, max_val=10)
            achievements = (talent_population ** weight_talent) * (effort_population ** weight_effort) * (luck_population ** weight_luck)
            grand_achievements.append(achievements)

        grand_achievements = np.sum(grand_achievements, axis=0)

        # Calculate Threshold for the Percentile
        threshold = np.percentile(grand_achievements, 100 - percentile)

        # Calculate User's Percentile Performance
        success_count = np.sum(user_results >= threshold)
        probability = (success_count / num_simulations) * 100

        # Display Results
        st.write(f"### Results for {attempt_label} Number of Attempts ({attempts} Attempts):")
        st.write(f"Likelihood of being in the Top {percentile}%: {probability:.2f}%")

