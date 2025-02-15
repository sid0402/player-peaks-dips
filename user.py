import streamlit as st
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Title for the Streamlit app
st.title("Cricket Player Performance Comparison")
st.write("Compare any two players based on their normalized 20-inning running averages and find similar players.")

# ----------------------------------
# Data Loading & Processing Function
# ----------------------------------
@st.cache_data
def load_data():
    if os.path.exists("running_players.pkl") and os.path.exists("career_average_players.pkl"):
        with open("running_players.pkl", "rb") as f:
            running_players = pickle.load(f)
        with open("career_average_players.pkl", "rb") as f:
            career_average_players = pickle.load(f)
    else:
        excel_file = '/Users/siddhantagarwal/Desktop/Ananth_Test_Database.xlsx'
        df = pd.read_excel(excel_file, sheet_name=1)
        df.columns = df.columns.str.strip()
        
        running_players = {}
        career_average_players = {}
        for i, row in df.iterrows():
            player = row['Player-Name'].strip()
            if player not in career_average_players:
                career_average_players[player] = {
                    'innings': 0,
                    'runs_scored': 0,
                    'balls_faced': 0,
                    'times_out': 0,
                }
            if player not in running_players:
                running_players[player] = []
            
            # Start a new 20-innings stretch if needed.
            if career_average_players[player]['innings'] % 20 == 0:
                running_players[player].append({'runs': 0, 'times_out': 0})
            
            if row['Balls'] == 0:
                continue
            
            career_average_players[player]['innings'] += 1
            career_average_players[player]['runs_scored'] += row['Runs']
            career_average_players[player]['balls_faced'] += row['Balls']
            running_players[player][-1]['runs'] += row['Runs']
            if row['NO'].strip() == '':
                career_average_players[player]['times_out'] += 1
                running_players[player][-1]['times_out'] += 1
        
        with open("running_players.pkl", "wb") as f:
            pickle.dump(running_players, f)
        with open("career_average_players.pkl", "wb") as f:
            pickle.dump(career_average_players, f)
    
    player_df = pd.DataFrame(career_average_players).transpose()
    player_df['batting_average'] = player_df['runs_scored'] / player_df['times_out']
    
    player_df = player_df[player_df['runs_scored'] >= 5000]
    player_df = player_df.sort_values(by='runs_scored', ascending=False)
    player_dict = player_df.to_dict(orient="index")
    
    running_avg_norm = {}
    for player in player_dict.keys():
        if player in running_players:
            stretches = running_players[player]
            averages = []
            for stretch in stretches:
                if stretch['times_out'] == 0:
                    continue
                averages.append(stretch['runs'] / stretch['times_out'])
            if len(averages) == 0:
                continue
            career_avg = player_dict[player]['batting_average']
            averages_norm = [avg / career_avg for avg in averages]
            running_avg_norm[player] = averages_norm
    return player_dict, running_avg_norm

player_dict, running_avg_norm = load_data()
all_players = sorted(list(running_avg_norm.keys()))

# ---------------------------
# Section 1: Compare Two Players
# ---------------------------
st.sidebar.header("Compare Two Players")
player1 = st.sidebar.selectbox("Select Player 1", all_players, index=0)
player2 = st.sidebar.selectbox("Select Player 2", all_players, index=1)

if player1 and player2:
    st.header(f"Comparison: {player1} vs {player2}")
    series1 = running_avg_norm[player1]
    series2 = running_avg_norm[player2]
    
    num_patches = min(len(series1), len(series2))
    np_series1 = np.array([[i, series1[i]] for i in range(num_patches)])
    np_series2 = np.array([[i, series2[i]] for i in range(num_patches)])
    dtw_distance, _ = fastdtw(np_series1, np_series2, dist=euclidean)
    st.write(f"**DTW distance** between {player1} and {player2}: `{dtw_distance:.4f}`")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x1 = [20 * (i + 1) for i in range(len(series1))]
    x2 = [20 * (i + 1) for i in range(len(series2))]
    ax.plot(x1, series1, marker='o', label=f"{player1} (Avg: {player_dict[player1]['batting_average']:.2f})")
    ax.plot(x2, series2, marker='o', label=f"{player2} (Avg: {player_dict[player2]['batting_average']:.2f})")
    
    ax.axhline(y=1, color='red', linestyle='--', label="Normalized Average = 1")
    ax.set_xlabel("Innings")
    ax.set_ylabel("Normalized Running Average")
    ax.set_title("20-Inning Running Average (Normalized by Career Average)")
    ax.legend()
    st.pyplot(fig)

# ---------------------------
# Section 2: Find Most Similar Players
# ---------------------------
st.sidebar.header("Find Similar Players")
base_player = st.sidebar.selectbox("Select a Base Player", all_players, key="base_player")
if base_player:
    st.header(f"Players Most Similar to {base_player}")
    base_series = running_avg_norm[base_player]
    similarities = {}
    for other in all_players:
        if other == base_player:
            continue
        other_series = running_avg_norm[other]
        num_patches = min(len(base_series), len(other_series))
        np_base = np.array([[i, base_series[i]] for i in range(num_patches)])
        np_other = np.array([[i, other_series[i]] for i in range(num_patches)])
        dist, _ = fastdtw(np_base, np_other, dist=euclidean)
        similarities[other] = dist
    
    similar_sorted = sorted(similarities.items(), key=lambda x: x[1])
    
    st.write("### Top 5 Similar Players:")
    for other, dist in similar_sorted[:5]:
        st.write(f"- **{other}**: DTW distance = `{dist:.4f}`")
