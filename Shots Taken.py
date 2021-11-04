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
start_frame = 0/90

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

home_shots_array = []
away_shots_array = []

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
                event_types = [OptaEventTypes.Miss,OptaEventTypes.Save,OptaEventTypes.Post]
                team_id_in_event = event.team_id
                
                for i in event_types:
                    if event.type_id == i:
                        if team_id_in_event == home_id:
                            home_shots_array.append(frame.frame_id)
                        elif team_id_in_event == away_id:
                            away_shots_array.append(frame.frame_id)
                        break



x = range(number_of_frames)
home_shots_by_frame = []
away_shots_by_frame = []


for i in x:
    frame_contains_shot = False
    for frame_id in home_shots_array:
        if frame_id == i:
            home_shots_by_frame.append(1)
            frame_contains_shot = True
            break
        
    if not frame_contains_shot:
        home_shots_by_frame.append(0)
        
    frame_contains_shot = False
    for frame_id in away_shots_array:
        if frame_id == i:
            away_shots_by_frame.append(1)
            frame_contains_shot = True
            break
        
    if not frame_contains_shot:
        away_shots_by_frame.append(0)


plt.plot(x,home_shots_by_frame,color='b')
plt.plot(x,away_shots_by_frame,color='r')
plt.show()