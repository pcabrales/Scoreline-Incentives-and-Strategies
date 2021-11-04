# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:31:02 2021

@author: hughc
"""

from fbpy.gamepack import GamePackToken, TeamType, TrackingFrameRate
from fbpy.opta.enumerations import OptaEventTypes
import matplotlib.pyplot as plt
import numpy as np

source_directory = r"C:\Users\hughc\Dropbox\MPhys Project\Indicators\data" # if databricks, use "/dbfs/mnt/gamepacks"
season_id = "2020"  # can be integer too, function will convert to str
competition = "PremierLeague"  # the exact name in the competition folder
match_id = 2128372  # ID of match, the name of the gamepack directory
fps5 = TrackingFrameRate.FPS5  # selects which subdir to load (25fps or 5fps)
    
smooth_value = int(30*5)
show_goals = False

chunk_size = 2*5*60

# Create GamePack token.
gpack = GamePackToken(
    match_id,
    season_id,
    competition,
    base_filepath=source_directory)

def length_of_match(match):
    num = 0
    for frame in match:
        num+=1
        
    return num


# Create match object using gamepack token, this will load most 
# components of the GamePack.
fbmatch = gpack.create_match_from_this_token(fps=fps5, load_eventjoin=True)

'''find frame of goals'''

home_id = fbmatch.metadata.home_roster.team_id
away_id = fbmatch.metadata.away_roster.team_id

homegoals = []
awaygoals = []

home_m_arr = []
away_m_arr = []

# Iterate through the eventjoin:
for event in fbmatch.eventjoin:
    # Event is an EventJoinEvent object. It stores useful metadata about each event.

    # Check if event is a goal by checking the type_id of the event.
    if event.type_id == OptaEventTypes.Goal:
        # event is a goal event.
        # Check which team scored the event, by checking the team_id associated with the event.
        if event.team_id == home_id:
            homegoals.append(event.frame_id)
        elif event.team_id == away_id:
            awaygoals.append(event.frame_id)

'''find players behind ball'''

number_of_frames = length_of_match(fbmatch)
number_of_chunks = int(number_of_frames/chunk_size)

for chunk in range(number_of_chunks):
    start = chunk*chunk_size
    stop = (chunk+1)*chunk_size
        
    home_players_behind_ball_arr = []
    away_players_behind_ball_arr = []
    home_players_behind_ball_arr_smoothed = []
    away_players_behind_ball_arr_smoothed = []
    
    for frame in fbmatch:
        if frame.frame_id>=start and frame.frame_id<stop:
            if frame:
                home_players_behind_ball = 0
                away_players_behind_ball = 0
                
                ball_in_frame = frame.ball
                ball_x_pos = ball_in_frame.x_pos
                
                hometeam = frame.hometeam
                awayteam = frame.awayteam
        
                # iterate through hometeam
                for player in hometeam:
                    player_x_pos = player.x_pos
                    
                    if player_x_pos < ball_x_pos:
                        home_players_behind_ball+=1
                        
                for player in awayteam:
                    player_x_pos = player.x_pos
                    
                    if player_x_pos > ball_x_pos:
                        away_players_behind_ball+=1
                        
                home_players_behind_ball_arr.append(home_players_behind_ball)
                away_players_behind_ball_arr.append(away_players_behind_ball)
    
    
    for j in range(len(home_players_behind_ball_arr)):
        if j > smooth_value:
            home_players_behind_ball_arr_smoothed.append(np.average(home_players_behind_ball_arr[j-smooth_value:j]))
        else:
            home_players_behind_ball_arr_smoothed.append(np.average(home_players_behind_ball_arr[0:j+1]))
            
    for j in range(len(away_players_behind_ball_arr)):
        if j >= smooth_value:
            away_players_behind_ball_arr_smoothed.append(np.average(away_players_behind_ball_arr[j-smooth_value:j]))
        else:
            away_players_behind_ball_arr_smoothed.append(np.average(away_players_behind_ball_arr[0:j+1]))
    
    max_frame = len(away_players_behind_ball_arr)
    x = range(max_frame)
    
    try:
        m_home, c_home = np.polyfit(x[smooth_value:], home_players_behind_ball_arr_smoothed[smooth_value:], 1)
        m_away, c_away = np.polyfit(x[smooth_value:], away_players_behind_ball_arr_smoothed[smooth_value:], 1)
    
        home_m_arr.append(m_home)
        away_m_arr.append(m_away)
    except:
        pass
    
plt.scatter(range(len(home_m_arr)),home_m_arr,color='b')
plt.scatter(range(len(away_m_arr)),away_m_arr,color='r')

x = range(len(away_m_arr))
    
m, c = np.polyfit(range(len(home_m_arr)),home_m_arr, 1)
plt.plot(x, m * x + c,color='b',label='Home')

m, c = np.polyfit(range(len(away_m_arr)),away_m_arr, 1)
plt.plot(x, m * x + c,color='r',label='Away')

plt.legend()
plt.show()


'''
plt.plot(x, y,'--',color='b', linewidth=0.4, alpha=0.7)
plt.plot(x, y,'--',color='r', linewidth=0.4, alpha=0.7)

top = float(np.max([np.max(home_players_behind_ball_arr_smoothed[smooth_value:]),np.max(away_players_behind_ball_arr_smoothed[smooth_value:])]))
bottom = float(np.min([np.min(home_players_behind_ball_arr_smoothed[smooth_value:]),np.min(away_players_behind_ball_arr_smoothed[smooth_value:])]))

m, c = np.polyfit(x[smooth_value:], y[smooth_value:], 1)
plt.plot(x, m * x + c,color='r',label='Away')

plt.xlabel('Frame')
plt.ylabel('Players Beheind Ball')

plt.legend()
plt.ylim(bottom-0.1,top+0.1)
plt.show()
'''