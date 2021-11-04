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

show_goals=False
start_frame = 80/90

source_directory = r"C:\Users\hughc\Dropbox\MPhys Project\Indicators\data" # if databricks, use "/dbfs/mnt/gamepacks"
season_id = "2020"  # can be integer too, function will convert to str
competition = "PremierLeague"  # the exact name in the competition folder
match_id = 2128362  # ID of match, the name of the gamepack directory
fps5 = TrackingFrameRate.FPS5  # selects which subdir to load (25fps or 5fps)
    
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

# Create GamePack token.
gpack = GamePackToken(
    match_id,
    season_id,
    competition,
    base_filepath=source_directory)

# Create match object using gamepack token, this will load most 
# components of the GamePack.
fb_match = gpack.create_match_from_this_token(fps=fps5, load_eventjoin=True,quiet=True)

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
            homegoals.append(event.frame_id)
        elif event.team_id == away_id:
            awaygoals.append(event.frame_id)

n_events = 0
n_passes = 0
# we will store the events and end coordinates in these lists. 
# Not all pass events have end coordinates (not sure if Opta 
# error or there's a physical reason for this [e.g. last kick of the game])
events_with_end_coordinates = []
end_opta_coordinates_of_events = []
end_phys_coordinates_of_events = []

home_passes_array = []
away_passes_array = []

number_of_frames = 0
for frame in fb_match:
    number_of_frames+=1

#start_frame = homegoals[0]/number_of_frames
frame_starting_point = int(number_of_frames*start_frame)

number_of_alive_frames = 0    
# iterate through frames in the match
for frame in fb_match:
    if frame.frame_id>frame_starting_point:
        if frame:
            number_of_alive_frames+=1
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
                            home_passes_array.append([frame.frame_id,physical_coordinates])
                        elif team_id_in_event == away_id:
                            away_passes_array.append([frame.frame_id,physical_coordinates])


x = range(number_of_frames-frame_starting_point)
x1 = range(number_of_frames)
y1 = []
y2 = []
y1_full = []
y2_full = []

for i in x1:
    frame_contains_pass = False
    for j in home_passes_array:
        frame_id = j[0]
        if frame_id == i:
            x_cord = j[1][0]
            if x_cord<0:
                y1.append(-x_cord)
                y1_full.append(-x_cord)
                frame_contains_pass = True
                break
    
    if not frame_contains_pass and i>=frame_starting_point:
        y1.append(0)
    
    frame_contains_pass = False
    for j in away_passes_array:
        frame_id = j[0]
        if frame_id == i:
            x_cord = j[1][0]
            if x_cord>0:
                y2.append(x_cord)
                y2_full.append(x_cord)
                frame_contains_pass = True
                break
    
    if not frame_contains_pass and i>=frame_starting_point:
        y2.append(0)

plt.plot(x, y1,color='b', linewidth=1, alpha=0.2)
plt.plot(x, y2,color='r', linewidth=1, alpha=0.2)

m, c = np.polyfit(range(len(y1_full)), y1_full, 1)
plt.plot(x, m*(len(y1_full)/len(y1)) * x + c,color='b',label='Home')
m, c = np.polyfit(range(len(y2_full)), y2_full, 1)
plt.plot(x, m*(len(y2_full)/len(y2)) * x + c,color='r',label='Away')
plt.xlabel('Frame')
plt.legend()

if show_goals:
    for j in homegoals:
        plt.axvline(j, ymin=0, color='b',linestyle=':')
    for j in awaygoals:
        plt.axvline(j, ymin=0, color='r',linestyle=':')

plt.show()