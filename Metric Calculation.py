# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:31:02 2021

@author: hughc
"""

from fbpy.gamepack import (GamePackToken, Match, OptaEventTypes,
                           OptaF24QualifierTypes, TeamType, TrackingFrameRate)
from fbpy.opta.enumerations import OptaEventTypes
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import math

source_directory = r"C:\Users\hughc\OneDrive\Documents\Data" # if databricks, use "/dbfs/mnt/gamepacks"
season_id = "2020"  # can be integer too, function will convert to str
competition = "PremierLeague"  # the exact name in the competition folder
match_id = 2128370 # ID of match, the name of the gamepack directory
fps5 = TrackingFrameRate.FPS5  # selects which subdir to load (25fps or 5fps)
chunk_size_mins = 1
chunk_size = chunk_size_mins*60*5 #convert to frames
h = chunk_size_mins
include_dead_frames = False
ignore_possesion = False
time_start = 0


def convert_opta_coordinates_to_pitch_coordinates_5fps_ONLY(match: Match, opta_coordinates: List[float], team_id_of_event: int) -> List[int]:
    """Function to convert Opta coordinates into physical pitch coordinates. Only works with 5fps data.
    """
    pitch_x, pitch_y = match.metadata.pitch_dims
    opta_x, opta_y = opta_coordinates
    phys_x = round((opta_x / 100 - 0.5) * pitch_x)
    phys_y = round((opta_y / 100 - 0.5) * pitch_y)
    if team_id_of_event == match.metadata.away_roster.team_id:
        # event is from away team's perspective, we need to rotate the coordinates
        phys_x *= -1
        phys_y *= -1
    return [phys_x, phys_y]

def length_of_match(match):
    num = 0
    for frame in match:
        num+=1
        
    return num

def get_players_behind_ball(frame):
    home_players_behind_ball = 0
    away_players_behind_ball = 0
            
    ball_in_frame = frame.ball
    ball_x_pos = ball_in_frame.x_pos
            
    hometeam = frame.hometeam
    awayteam = frame.awayteam
    
    ball_possession = ball_in_frame.owning_team
    
    if ball_possession == TeamType.AWAY or ignore_possesion:

        for player in hometeam:
            player_x_pos = player.x_pos
            
            if player_x_pos < ball_x_pos:
                home_players_behind_ball+=1
    else:
        home_players_behind_ball = None
    
    if ball_possession == TeamType.HOME or ignore_possesion:     
        for player in awayteam:
            player_x_pos = player.x_pos
            
            if player_x_pos > ball_x_pos:
                away_players_behind_ball+=1
    else:
        away_players_behind_ball = None
            
    return [home_players_behind_ball,away_players_behind_ball]  
    
def calculate_metric(players_behind_ball_arr,pass_x_coords_arr):
    
    pass_not_none_array = []
    players_behind_ball_not_none_arr = []
    
    for j in pass_x_coords_arr:
        if j is not None:
            pass_not_none_array.append(((x_coords_max-x_coords_min)/2)+j)
            
    for j in players_behind_ball_arr:
        if j is not None:
            players_behind_ball_not_none_arr.append(j)
        
    A = 1/(x_coords_max-x_coords_min)
    B = 1/11
    players_behind_ball_average = np.average(players_behind_ball_not_none_arr)
    pass_cord_average = np.average(pass_not_none_array)
    
    players_behind_ball_contribution = B * (11-players_behind_ball_average)
    pass_average_contribution = 1-A*pass_cord_average
    #print(players_behind_ball_average)
    #print(pass_cord_average)
    
    if not (pass_not_none_array == [] or players_behind_ball_arr == []):
        metric = 0.5*(players_behind_ball_contribution + pass_average_contribution)
    elif players_behind_ball_arr == []:
        metric = pass_average_contribution
    else:
        metric = players_behind_ball_contribution
    #metric = (1-A*pass_cord_average)
    #metric = B * (11-players_behind_ball_average)
    
    return [metric,[pass_average_contribution,players_behind_ball_contribution]]
    
def chunks(arr, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(arr), n):
        yield arr[i:i + n] 

def integrate(x, y, x_start):
    x_start_index = int(len(x)*(x_start/np.max(x)))
    y = y[x_start_index:]
    x = x[x_start_index:]
    #print(x)
    #print(y)
    A = 0.5 * h * (y[0] + y[-1] + 2 * (np.sum(y[1:-1])))
    
    return A
    
# Create GamePack token.
gpack = GamePackToken(
    match_id,
    season_id,
    competition,
    base_filepath=source_directory)

# Create match object using gamepack token, this will load most 
# components of the GamePack.
fb_match = gpack.create_match_from_this_token(fps=fps5, load_eventjoin=True,quiet=True)
total_number_of_frames = length_of_match(fb_match)

home_id = fb_match.metadata.home_roster.team_id
away_id = fb_match.metadata.away_roster.team_id

homegoals = []
awaygoals = []

# Iterate through the eventjoin:
for event in fb_match.eventjoin:
    # Event is an EventJoinEvent object. It stores useful metadata about each event.

    # Check if event is a goal by checking the type_id of the event.
    if event.type_id == OptaEventTypes.Goal:
        # event is a goal event.
        # Check which team scored the event, by checking the team_id associated with the event.
        if event.team_id == home_id:
            homegoals.append(event.frame_id/total_number_of_frames)
        elif event.team_id == away_id:
            awaygoals.append(event.frame_id/total_number_of_frames)

    
n_events = 0
n_passes = 0

home_players_behind_ball_arr = []
away_players_behind_ball_arr = []

# we will store the events and end coordinates in these lists. 
# Not all pass events have end coordinates (not sure if Opta 
# error or there's a physical reason for this [e.g. last kick of the game])
events_with_end_coordinates = []
end_opta_coordinates_of_events = []
end_phys_coordinates_of_events = []

home_passes_array = []
away_passes_array = []


#start_frame = homegoals[0]/number_of_frames

for frame in fb_match:
    frame_id = frame.frame_id
    
    if frame or include_dead_frames:
        players_behind_ball_frame = get_players_behind_ball(frame)
        home_players_behind_ball_arr.append(players_behind_ball_frame[0])
        away_players_behind_ball_arr.append(players_behind_ball_frame[1])
        
        pass_appended = False
        # frame is a GamePack.Frame object
        for event in frame.events:
            n_events += 1
            # select event if it's a pass
            if event.type_id == OptaEventTypes.Pass:
                n_passes += 1
                opta_event = event.corresponding_opta_event
                pass_end_x = None
                pass_end_y = None
                for qualifier in opta_event: 
                    # check qualifier ID and see if its the desired type
                    if qualifier.qualifier_id == OptaF24QualifierTypes.PassEndX:
                        # note that qualifier values are stored as strings 
                        # (they contain various types, so string is most versatile)
                        pass_end_x = qualifier.value 
                    elif qualifier.qualifier_id == OptaF24QualifierTypes.PassEndY:
                        pass_end_y = qualifier.value
                # check that both qualifiers were found
                if pass_end_x is not None and pass_end_y is not None:
                    # store event
                    events_with_end_coordinates.append(event)
                    # Cast coordinates to floats and store them
                    opta_coordinates = [float(pass_end_x), float(pass_end_y)]
                    end_opta_coordinates_of_events.append(opta_coordinates)
                    # be warned that these are Opta coordinates and must be converted back to pitch coordinates
                    team_id_in_event = event.team_id
                    physical_coordinates = convert_opta_coordinates_to_pitch_coordinates_5fps_ONLY(fb_match, opta_coordinates, team_id_in_event)
                    
                    if team_id_in_event == home_id:
                        home_passes_array.append([frame_id,physical_coordinates])
                        pass_appened = True
                    elif team_id_in_event == away_id:
                        away_passes_array.append([frame_id,physical_coordinates])
                        pass_appened = True
                        
        if not pass_appended:
            home_passes_array.append(None)
            away_passes_array.append(None)
            
    else:
        home_players_behind_ball_arr.append(None)
        away_players_behind_ball_arr.append(None)
        home_passes_array.append(None)
        away_passes_array.append(None)
        
x_coords_array_test=[]
home_x_coords_array = []
away_x_coords_array = []

for pass_event in home_passes_array:
    if pass_event is not None:
        x_coord = -pass_event[1][0]
        x_coords_array_test.append(x_coord)
        home_x_coords_array.append(x_coord)
    else:
        home_x_coords_array.append(None)
for pass_event in away_passes_array:
    if pass_event is not None:
        x_coord = pass_event[1][0]
        x_coords_array_test.append(x_coord)
        away_x_coords_array.append(x_coord)
    else:
        away_x_coords_array.append(None)
    
    
x_coords_min = np.min(x_coords_array_test)
x_coords_max = np.max(x_coords_array_test)

chunk_ranges = chunks(range(total_number_of_frames),chunk_size)
chunk_ranges = list(chunk_ranges)
chunk_ranges_temp = chunk_ranges[0:-2]
chunk_ranges_temp.append(range(chunk_ranges[-2][0],chunk_ranges[-1][-1]))
chunk_ranges=chunk_ranges_temp

metric_array_home = []
metric_array_away = []

home_contributions_array = []
away_contributions_array = []

for chunk_range in chunk_ranges:
            
    home_passes_chunk=home_x_coords_array[chunk_range[0]:chunk_range[-1]]
    away_passes_chunk=away_x_coords_array[chunk_range[0]:chunk_range[-1]]
    home_players_behind_ball_chunk=home_players_behind_ball_arr[chunk_range[0]:chunk_range[-1]]
    away_players_behind_ball_chunk=away_players_behind_ball_arr[chunk_range[0]:chunk_range[-1]]
    
    home_metric_result = calculate_metric(home_players_behind_ball_chunk,home_passes_chunk)
    away_metric_result = calculate_metric(away_players_behind_ball_chunk,away_passes_chunk)
    
    if not math.isnan(home_metric_result[0]):
        home_metric = home_metric_result[0]
    if not math.isnan(away_metric_result[0]):
        away_metric = away_metric_result[0]
    
    home_contributions = home_metric_result[1]
    away_contributions = away_metric_result[1]
    
    home_contributions_array.append(home_contributions)
    away_contributions_array.append(away_contributions)
    
    metric_array_home.append(home_metric)
    metric_array_away.append(away_metric)
    
x1 = range(int(total_number_of_frames/chunk_size))
x1_max = x1[-1]

max_time = total_number_of_frames/(5*60)
time_factor = int((max_time)/x1_max)
x = np.multiply(x1,time_factor)
x_max = x[-1]

x_plot = np.add(x,chunk_size_mins/2)
#max_time=90
#plt.bar(x-2+chunk_size_mins,metric_array_home,width=2,align='center',color='b')
#plt.bar(x+2+chunk_size_mins,metric_array_away,width=2,align='center',color='r')


home_contributions_array = np.array(home_contributions_array)
away_contributions_array = np.array(away_contributions_array)

'''
for j in range(2):
    plt.plot(x_plot,home_contributions_array[:,j],color='b',alpha = 0.5,linestyle=':')
    plt.plot(x_plot,away_contributions_array[:,j],color='r',alpha = 0.5,linestyle=':')
'''  

h_integrate_array = []
a_integrate_array = []

if x_max==90:
    range_temp = range(time_start,90,chunk_size_mins)
else:
    range_temp = range(time_start,90+chunk_size_mins,chunk_size_mins)
    
for i in range_temp:
    x_start = i
    h_integrate_array.append(integrate(x,metric_array_home,x_start)/(x_max-x_start))
    a_integrate_array.append(integrate(x,metric_array_away,x_start)/(x_max-x_start))
    

range_temp = np.add(range_temp,chunk_size_mins/2)

for j in homegoals:
    plt.axvline(j*x_max, ymin=0, color='b',linestyle=':')
                
for j in awaygoals:
    plt.axvline(j*x_max, ymin=0, color='r',linestyle=':')

plt.plot(x_plot,metric_array_home,color='b',label='Home')
plt.plot(x_plot,metric_array_away,color='r',label='Away')
plt.legend()
plt.grid(axis='x')
plt.xlabel('Time (min)')
plt.ylabel('Metric')
plt.savefig('Metric.png',dpi=800)
plt.show()


plt.plot(range_temp,h_integrate_array,color='b',label='Home')
plt.plot(range_temp,a_integrate_array,color='r',label='Away')
plt.xlabel('Time (min)')
plt.ylabel('Integrated Metric')
plt.grid(axis='x')
plt.legend()
for j in homegoals:
    if j*x_max>time_start:
        plt.axvline(j*x_max, ymin=0, color='b',linestyle=':')
                
for j in awaygoals:
    if j*x_max>time_start:
        plt.axvline(j*x_max, ymin=0, color='r',linestyle=':')
        
plt.savefig('IntegratedMetric.png',dpi=800)
plt.show()

    