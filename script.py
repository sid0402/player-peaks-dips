import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

f = '/Users/siddhantagarwal/Desktop/Ananth_Test_Database.xlsx'
#df = pd.read_excel(f, sheet_name=1)

#df.columns = df.columns.str.strip()

def find_running_players():
    running_players = {}
    career_average_players = {}
    for i, row in df.iterrows():
        print(row['Player-Name'])
        player = row['Player-Name'].strip()
        if player not in career_average_players:
            career_average_players[player] = {
                'innings':0,
                'runs_scored':0,
                'balls_faced':0,
                'times_out':0,        
            }

        if player not in running_players:
            running_players[player] = []

        if career_average_players[player]['innings'] % 20 == 0:
            running_players[player].append({'runs':0, 'times_out':0})
        
        if row['Balls'] == 0:
            continue
        career_average_players[player]['innings']+=1
        career_average_players[player]['runs_scored']+=row['Runs']
        career_average_players[player]['balls_faced']+=row['Balls']
        running_players[player][-1]['runs']+=row['Runs']
        if row['NO'].strip() == '':
            career_average_players[player]['times_out']+=1
            running_players[player][-1]['times_out']+=1

    with open('running_players.pkl','wb') as f:
            pickle.dump(running_players, f)
            f.close()

    with open('career_average_players.pkl','wb') as f:
            pickle.dump(career_average_players, f)
            f.close()

#find_running_players()

with open('running_players.pkl','rb') as f:
    running_players = pickle.load(f)
    f.close()

with open('career_average_players.pkl','rb') as f:
    career_average_players = pickle.load(f)
    f.close()

player_df = pd.DataFrame(career_average_players).transpose()
player_df['batting_average'] = player_df['runs_scored'] / player_df['times_out']
player_df = player_df[player_df['runs_scored']>=8000]
player_df = player_df.sort_values(by='runs_scored', ascending=False)
player_dict = player_df.to_dict(orient='index')

chosen_players = player_dict.keys()
chosen_players = ['V Kohli','JH Kallis']
chosen_running = {key: running_players[key] for key in chosen_players}

chosen_running_averages = {}
chosen_running_averages_normalized = {}

for player, stretches in chosen_running.items():
    averages = []
    for i,d in enumerate(stretches):
        if d['times_out']==0:
            continue
        else:
            averages.append(d['runs']/d['times_out'])
    career_average = player_dict[player]['batting_average']
    averages_normalized = [average/career_average for average in averages]
    chosen_running_averages[player] = averages
    chosen_running_averages_normalized[player] = averages_normalized

players = list(chosen_running_averages_normalized.keys())

'''
pairwise_similarities = {}
for i in range(len(players)):
    p1, p2 = 'V Kohli', players[i]
    #series1, series2 = chosen_running_averages_normalized[p1], chosen_running_averages_normalized[p2]
    series1, series2 = chosen_running_averages_normalized[p1], chosen_running_averages_normalized[p2]
    #num_patches = min(len(series1), len(series2))
    num_patches = 10
    npseries1 = np.array([[i, ave]for i, ave in enumerate(series1) if i<num_patches])
    npseries2 = np.array([[i, ave] for i, ave in enumerate(series2) if i <num_patches])
    distance, _ = fastdtw(npseries1, npseries2, dist=euclidean)
    #pairwise_similarities[(p1, p2)] = distance * abs(player_dict[p1]['batting_average'] - player_dict[p2]['batting_average'])**0.25
    pairwise_similarities[(p1, p2)] = distance

sorted_pairs = sorted(pairwise_similarities.items(), key=lambda x: x[1])

print("Pairwise DTW distances between players (lower = more similar):")
for i in range(min(20, len(sorted_pairs))):
    print(f"{sorted_pairs[i][0]}: {sorted_pairs[i][1]:.4f}")
'''

peaks = {i:[] for i in range(12)}
for player in players:
    for i,av in enumerate(chosen_running_averages_normalized[player]):
        if i == 12:
            break
        peaks[i].append(av)

peaks = {i:np.mean(av) for i, av in peaks.items() if av}
print(peaks)
    

player_stds = {}
for player in players:
    player_stds[player] = np.std(chosen_running_averages_normalized[player])

#print(sorted(player_stds.items(), key=lambda d: d[1]))

plt.figure(figsize=(12, 8))
num_players = len(chosen_running_averages_normalized)
colors = plt.cm.tab20(np.linspace(0, 1, num_players))
'''
x_values = [(i+1)*20 for i in range(12)]
values = list(peaks.values())
plt.plot(x_values, values, marker='o')
plt.xlabel('Innings')
plt.ylabel('Normalized Running Average')
plt.title("20-inning Running Average (normalized by player's career average)")
#plt.ylim(0, 2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
#plt.show()
'''

plt.figure(figsize=(12, 8))
# Iterate over each player and plot their data with a unique color.
for idx, (player, values) in enumerate(chosen_running_averages_normalized.items()):
    # x-values: each data point corresponds to 20*(i+1) innings.
    x_values = [20 * (i + 1) for i in range(len(values))]
    plt.plot(x_values, values, marker='o', label=f'{player} - Career Average: {player_dict[player]['batting_average']:.2f}', color=colors[idx])
plt.axhline(y=1, color='red', linestyle='--', label='Normalized running average = 1')
plt.xlabel('Innings')
plt.ylabel('Normalized Running Average')
plt.title("20-inning Running Average (normalized by player's career average)")
#plt.ylim(0, 2)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))

# Iterate over each player and plot their data with a unique color.
for idx, (player, values) in enumerate(chosen_running_averages_normalized.items()):
    # x-values: each data point corresponds to 20*(i+1) innings.
    x_values = [20 * (i + 1) for i in range(len(values))]
    plt.plot(x_values, values, marker='o', label=f'{player} - {player_dict[player]['batting_average']:.2f}', color=colors[idx])
plt.xlabel('Innings')
plt.ylabel('Normalized Running Average')
plt.title("20-inning Running Average (normalized by player's career average)")
#plt.ylim(0, 2)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

result_players = []
for player, values in chosen_running_averages_normalized.items():
    if any(v > 1.25 for v in values[11:]):
        result_players.append((player))
print("Players with any value > 1.25 after the 11th entry:", result_players)


'''
['SR Tendulkar', 'RT Ponting', 'JH Kallis', 'R Dravid', 'JE Root', 'AN Cook', 'KC Sangakkara', 'BC Lara', 'S Chanderpaul', 'M Jayawardene', 'AR Border', 'SR Waugh', 'SPD Smith', 'Younis Khan', 'HM Amla', 'KS Williamson', 'GC Smith', 'V Kohli', 'GA Gooch']
'''