# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:31:02 2021

@author: hughc
"""

from fbpy.gamepack import GamePackToken, TeamType, TrackingFrameRate
from fbpy.opta.enumerations import OptaEventTypes
import matplotlib.pyplot as plt
import numpy as np

source_directory = r"C:\Users\hughc\OneDrive\Documents\Data" # if databricks, use "/dbfs/mnt/gamepacks"
season_id = "2020"  # can be integer too, function will convert to str
competition = "PremierLeague"  # the exact name in the competition folder
match_id = 2128372 # ID of match, the name of the gamepack directory
fps5 = TrackingFrameRate.FPS5  # selects which subdir to load (25fps or 5fps)
    
smooth_value = 3*60*5
start_frame = 0/90
show_goals = True

# Create GamePack token.
gpack = GamePackToken(
    match_id,
    season_id,
    competition,
    base_filepath=source_directory)

# Create match object using gamepack token, this will load most 
# components of the GamePack.
fbmatch = gpack.create_match_from_this_token(fps=fps5, load_eventjoin=True,quiet=True)

'''find frame of goals'''

home_id = fbmatch.metadata.home_roster.team_id
away_id = fbmatch.metadata.away_roster.team_id

homegoals = []
awaygoals = []

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

home_players_behind_ball_arr = []
away_players_behind_ball_arr = []

number_of_frames = 0
for frame in fbmatch:
    number_of_frames+=1
    
for frame in fbmatch:
    if frame.frame_id>int(start_frame*number_of_frames):
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

home_players_behind_ball_arr_smoothed = []
away_players_behind_ball_arr_smoothed = []

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
y = home_players_behind_ball_arr_smoothed

plt.plot(x, y,'--',color='b', linewidth=0.4, alpha=0.7)
m, c = np.polyfit(x[smooth_value:], y[smooth_value:], 1)
plt.plot(x, m * x + c,color='b',label='Home')

if show_goals:
    for j in homegoals:
        plt.axvline(j, ymin=0, color='b',linestyle=':')
                
    for j in awaygoals:
        plt.axvline(j, ymin=0, color='r',linestyle=':')
            

y = away_players_behind_ball_arr_smoothed

top = float(np.max([np.max(home_players_behind_ball_arr_smoothed[smooth_value:]),np.max(away_players_behind_ball_arr_smoothed[smooth_value:])]))
bottom = float(np.min([np.min(home_players_behind_ball_arr_smoothed[smooth_value:]),np.min(away_players_behind_ball_arr_smoothed[smooth_value:])]))

plt.plot(x, y,'--',color='r', linewidth=0.4, alpha=0.7)
m, c = np.polyfit(x[smooth_value:], y[smooth_value:], 1)
plt.plot(x, m * x + c,color='r',label='Away')

plt.xlabel('Frame')
plt.ylabel('Players Beheind Ball')

plt.legend()
plt.ylim(bottom-0.1,top+0.1)
plt.show()