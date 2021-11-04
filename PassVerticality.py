import os
from typing import List
from scipy.stats import norm
import matplotlib.cm as cm
import math as math
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from fbpy.gamepack import (GamePackToken, Match, OptaEventTypes,
                           OptaF24QualifierTypes, TeamType, TrackingFrameRate)
from fbpy.opta.enumerations import OptaEventTypes
from fbpy.pitchcontrol import MatchProcessor
import fbpy as fbpy
import pandas as pd
from fbpy.core.enumerations import PeriodID, PlayerID
from fbpy.statsbomb.enumerations import SBEventTypeID, SBPositionID
from fbpy.statsbomb import loadfile_events, SBEventTypeID



#Converting opta coordinates to pitch coordinates
def convert_opta_coordinates_to_pitch_coordinates_5fps_ONLY(match: Match, opta_coordinates: List[float], team_id_of_event: int) -> List[int]:
    """Function to convert Opta coordinates into physical pitch coordinates. Only works with 5fps data.
    """
    pitch_x, pitch_y = match.metadata.pitch_dims
    opta_x, opta_y = opta_coordinates
    phys_x = round((opta_x / 100 - 0.5) * pitch_x)
    phys_y = round((opta_y / 100 - 0.5) * pitch_y)
    if team_id_of_event == match.metadata.away_roster.team_id:
        phys_x *= -1
        phys_y *= -1
    return [phys_x, phys_y]

# Importing match data:
source_directory = "data" 
season_id = "2020"  
competition = "PremierLeague"  
match_id = 2128362  
fps5 = TrackingFrameRate.FPS5 


# Create GamePack token:
gpack = GamePackToken(
    match_id,
    season_id,
    competition,
    base_filepath=source_directory)

# Create match object using gamepack token, this will load most 
# components of the GamePack:
fbmatch = gpack.create_match_from_this_token(fps=fps5, load_eventjoin=True)
home_id = fbmatch.metadata.home_roster.team_id
away_id = fbmatch.metadata.away_roster.team_id


#Predefining variables and arrays:
n_passes = 0

events_with_end_coordinates = []
end_opta_coordinates_of_events = []
end_phys_coordinates_of_events = []

pass_start_x_array = np.array([],'float')
pass_end_x_array = np.array([],'float')



###TEST STATSBOMB
##csv_path = ("/data/2020/PremierLeague/2128362/5fps/2128362.EVENTJOIN")
##
##sbevents = loadfile_events(csv_path)
##
##pass_count = 0
##for event in sbevents:
##    event_type = event.get_type_id()
##
##    if event_type == SBEventTypeID.PASS:
##        pass_count += 1
##        continue
##print (pass_count)
    
            

frame_goals = np.empty(shape=[0,2]) # First column is the frame of the goal
                                      # second column is the id of the team which scored
frame_passes = np.empty(shape=[0,3]) # First column is the frame of the goal
                                       # second column is the id of the team which scored
                                       # third column is the length of the pass
                                    
                                       
#Recording the frame in which each goal is scored and by which team:
n_frames=0
for frame in fbmatch:
    n_frames+=1
    for event in frame.events:
        if event.type_id == OptaEventTypes.Goal:
            frame_goals = np.vstack((frame_goals, [int(frame.frame_id),int(event.team_id)]))
        if event.type_id == OptaEventTypes.Pass:
            opta_event = event.corresponding_opta_event
            pass_end_x = None
            pass_end_y = None
            for qualifier in opta_event:
                if qualifier.qualifier_id == OptaF24QualifierTypes.PassEndX:
                    pass_end_x = qualifier.value
                elif qualifier.qualifier_id == OptaF24QualifierTypes.PassEndY:
                    pass_end_y = qualifier.value
            if pass_end_x is not None and pass_end_y is not None:
                opta_coordinates = [float(pass_end_x), float(pass_end_y)]
                physical_coordinates = convert_opta_coordinates_to_pitch_coordinates_5fps_ONLY(fbmatch, opta_coordinates, event.team_id)
                pass_end_x_array = np.append(pass_end_x_array, physical_coordinates[0]/100)
                pass_start_x_array = np.append(pass_start_x_array, event.event_physical_pos_x/100)
                frame_passes=np.vstack((frame_passes,np.array([int(frame.frame_id),(physical_coordinates[0]-event.event_physical_pos_x)/100,int(event.team_id)])))
                    


frame_passes_home = np.delete(frame_passes[frame_passes[:,2]==away_id],(2), axis=1)
frame_passes_away = np.delete(frame_passes[frame_passes[:,2]==home_id],(2), axis=1)


plt.figure()
plt.grid()
plt.ylabel('\u0394 x (m)')
plt.xlabel('Minutes')
#plt.xlim([0,50])
plt.title('Pass length')
plt.plot(frame_passes_home[:,0]/60/5,frame_passes_home[:,1],'b.')
plt.plot(frame_passes_away[:,0]/60/5,frame_passes_away[:,1],'r.')

for goal in frame_goals:
    if int(goal[1]) == home_id:
        plt.axvline(x=goal[0]/60/5,color='b', label ='Home team scores')
    else:
        plt.axvline(x=goal[0]/60/5,color='r', label ='Away team scores')

#We'll find the first pass after each goal:
#First goal:
index_frame_home_pass_after_first_goal = (np.absolute(frame_passes_home[:,0]-frame_goals[0,0])).argmin()        
index_frame_away_pass_after_first_goal = (np.absolute(frame_passes_away[:,0]-frame_goals[0,0])).argmin()

index_frame_home_pass_after_second_goal = (np.absolute(frame_passes_home[:,0]-frame_goals[1,0])).argmin()        
index_frame_away_pass_after_second_goal = (np.absolute(frame_passes_away[:,0]-frame_goals[1,0])).argmin()

pass_verticality_home = frame_passes_home[:,1]
pass_verticality_away = frame_passes_away[:,1]

pass_verticality_home_before_goal = frame_passes_home[0:index_frame_home_pass_after_first_goal-1,1]
pass_verticality_away_before_goal = frame_passes_away[0:index_frame_away_pass_after_first_goal-1,1]
pass_verticality_home_after_goal = frame_passes_home[index_frame_home_pass_after_first_goal-1:index_frame_home_pass_after_second_goal-1,1]
pass_verticality_away_after_goal = frame_passes_away[index_frame_away_pass_after_first_goal-1:index_frame_away_pass_after_second_goal-1,1]





##plt.axhline(y = np.average(pass_verticality_home_before_goal),xmin = 0, xmax=frame_goals[0,0]/60/5, label='Home pass verticality before the goal')
##plt.axhline(y = np.average(pass_verticality_away_before_goal),xmin = 0, xmax=frame_goals[0,0]/60/5, label='Away pass verticality before the goal')
##plt.axhline(y = np.average(pass_verticality_home_after_goal),xmin = frame_goals[0,0]/60/5, xmax=frame_goals[1,0]/60/5, label='Home pass verticality after the goal')
##plt.axhline(y = np.average(pass_verticality_away_after_goal),xmin = frame_goals[0,0]/60/5, xmax=frame_goals[1,0]/60/5, label='Away pass verticality after the goal')

plt.legend()

if frame_goals[0,1] == home_id:
    print('The home team scored the first goal.\n')
else:
    print('The away team scored the first goal.\n')
    
print('The average home verticality \u0394 x of a pass before the goal was:')
print('%.2f m'% (np.average(pass_verticality_home_before_goal)))
print('And, after the goal: ')
print('%.2f m'% (np.average(pass_verticality_home_after_goal)))

print()

print('The average away verticality \u0394 x of a pass before the goal was:')
print('%.2f m'% (np.average(pass_verticality_away_before_goal)))
print('And, after the goal: ')
print('%.2f m'% (np.average(pass_verticality_away_after_goal)))


##fit_before,V_before=np.polyfit(passes_before_goal, pass_verticality_before_goal, 1, cov=True)
##fit_after,V_after=np.polyfit(passes_after_goal, pass_verticality_after_goal, 1, cov=True)
##plt.plot(passes_before_goal,passes_before_goal*fit_before[0]+fit_before[1],'b')
##plt.plot(passes_after_goal,passes_after_goal*fit_after[0]+fit_after[1],'b')
##plt.legend()


print(f"Total number of_passes: {n_passes}")



#Supposing the longest pass (105 m, the whole pitch) comprises 4 standard deviations:
indicatorvalue_home_before_goal = norm.cdf(np.average(pass_verticality_home_before_goal*4/105))
indicatorvalue_home_after_goal = norm.cdf(np.average(pass_verticality_home_after_goal*4/105))

indicatorvalue_away_before_goal = norm.cdf(np.average(pass_verticality_away_before_goal*4/105))
indicatorvalue_away_after_goal = norm.cdf(np.average(pass_verticality_away_after_goal*4/105))

print()
print('Offensiveness value (from 0 to 1): ')
print('Home team:')
print("Before the goal: %f "%(indicatorvalue_home_before_goal))
print("After the goal: %f "%(indicatorvalue_home_after_goal))
print()
print('Away team:')
print("Before the goal: %f "%(indicatorvalue_away_before_goal))
print("After the goal: %f "%(indicatorvalue_away_after_goal))



##plt.figure(1)
##plt.title('Histogram of passes before and after conceding (normalised)')
##plt.xlabel('\u0394 x (m)')
##plt.ylabel('Number of passes')
##plt.hist(x = pass_verticality_before_goal, bins='auto', alpha=0.5, rwidth=0.85,label='Before conceding')
##plt.hist(x = pass_verticality_after_goal, bins='auto', alpha=0.5, rwidth=0.85,label='After conceding')
##plt.legend()


plt.figure()
plt.title('Histogram of HOME team passes before and after the goal (normalised)')
plt.xlabel('\u0394 x (m)')
num_bin = 20
bin_lims = np.linspace(np.amin(pass_verticality_home_before_goal),np.amax(pass_verticality_home_before_goal),num_bin+1)
bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
bin_widths = bin_lims[1:]-bin_lims[:-1]


hist1, _ = np.histogram(pass_verticality_home_before_goal, bins=bin_lims)
hist2, _ = np.histogram(pass_verticality_home_after_goal, bins=bin_lims)

hist1b = hist1/np.max(hist1)
hist2b = hist2/np.max(hist2)

plt.bar(bin_centers, hist1b, width = bin_widths*0.8, align = 'center',alpha=0.9,label='Before conceding')
plt.bar(bin_centers, hist2b, width = bin_widths*0.8, align = 'center',alpha = 0.5,label='After conceding')
plt.legend()



plt.figure()
plt.title('Histogram of AWAY team passes before and after the goal (normalised)')
plt.xlabel('\u0394 x (m)')
num_bin = 20
bin_lims = np.linspace(np.amin(pass_verticality_away_before_goal),np.amax(pass_verticality_away_before_goal),num_bin+1)
bin_centers = 0.5*(bin_lims[:-1]+bin_lims[1:])
bin_widths = bin_lims[1:]-bin_lims[:-1]


hist1, _ = np.histogram(pass_verticality_away_before_goal, bins=bin_lims)
hist2, _ = np.histogram(pass_verticality_away_after_goal, bins=bin_lims)

hist1b = hist1/np.max(hist1)
hist2b = hist2/np.max(hist2)

plt.bar(bin_centers, hist1b, width = bin_widths*0.8, align = 'center',alpha=0.9,label='Before conceding')
plt.bar(bin_centers, hist2b, width = bin_widths*0.8, align = 'center',alpha = 0.5,label='After conceding')
plt.legend()

##print('Maximum pass forward length: %.2f m'%(np.amax(pass_end_x_array)/100))
##print('Maximum pass backwards length %.2f'%(np.amin(pass_end_x_array)/100))

plt.figure()
plt.grid()
mu = 0
sigma = 105/4
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
plt.xlim([mu - 4*sigma, mu + 4*sigma])
plt.ylim([0,np.amax(stats.norm.pdf(x, mu, sigma))*1.1])
plt.xlabel('\u0394 x (m)')
plt.title('Distribution of the pass verticality as a gaussian function')
x_home_before = np.linspace(mu - 4*sigma, np.average(pass_verticality_home_before_goal), 1000)
x_home_after = np.linspace(mu - 4*sigma, np.average(pass_verticality_home_after_goal), 1000)

x_away_before = np.linspace(mu - 4*sigma, np.average(pass_verticality_away_before_goal), 1000)
x_away_after = np.linspace(mu - 4*sigma, np.average(pass_verticality_away_after_goal), 1000)

plt.fill_between(x_home_before ,stats.norm.pdf(x_home_before , mu, sigma),0, alpha=0.3, color='purple')
plt.fill_between(x_home_after ,stats.norm.pdf(x_home_after, mu, sigma),0, alpha=0.2, color='brown')

plt.fill_between(x_away_before ,stats.norm.pdf(x_away_before , mu, sigma),0, alpha=0.3, color='orange')
plt.fill_between(x_away_after ,stats.norm.pdf(x_away_after, mu, sigma),0, alpha=0.2, color='green')


plt.plot(x, stats.norm.pdf(x, mu, sigma))
if frame_goals[0,1] == home_id:
    plt.plot(np.average(pass_verticality_home_before_goal), stats.norm.pdf(np.average(pass_verticality_home_before_goal), mu, sigma),
             color='purple', marker = 'o', markersize = '12', label='Home before conceding')
    plt.plot(np.average(pass_verticality_home_after_goal), stats.norm.pdf(np.average(pass_verticality_home_after_goal), mu, sigma),
             color='brown', marker  = 'o', markersize = '12', label='Home after conceding')

elif frame_goals[0,1] == away_id:
    plt.plot(np.average(pass_verticality_away_before_goal), stats.norm.pdf(np.average(pass_verticality_away_before_goal), mu, sigma),
             color='orange', marker = 'o', markersize = '12', label='Away before conceding')
    plt.plot(np.average(pass_verticality_away_after_goal), stats.norm.pdf(np.average(pass_verticality_away_after_goal), mu, sigma),
             color='green', marker  = 'o', markersize = '12', label='Away after conceding')


plt.legend()

# CONTINUOUS INDICATOR VALUE
pass_verticality_home_normalised = pass_verticality_home/105*4
indicatorvalue_home = np.array([])
for pass_ in pass_verticality_home_normalised:
    indicatorvalue_home = np.append(indicatorvalue_home, norm.cdf(pass_))
    
pass_verticality_away_normalised = pass_verticality_away/105*4
indicatorvalue_away = np.array([])
for pass_ in pass_verticality_away_normalised:
    indicatorvalue_away = np.append(indicatorvalue_away, norm.cdf(pass_))


plt.figure()



plt.grid()
for goal in frame_goals:
    if int(goal[1]) == home_id:
        plt.axvline(x=goal[0]/60/5,color='b', label ='Home team scores')
    else:
        plt.axvline(x=goal[0]/60/5,color='r', label ='Away team scores')

##min_smooth=1
##smooth_value = int(min_smooth*60*5)
smooth_value = 15 #smoothing over the last smooth_value passes
indicatorvalue_home_smooth = np.array([])
indicatorvalue_away_smooth = np.array([])

for j in range(len(indicatorvalue_home)):
    if j > smooth_value:
        indicatorvalue_home_smooth=np.append(indicatorvalue_home_smooth,np.average(indicatorvalue_home[j-smooth_value:j]))
    else:
        indicatorvalue_home_smooth=np.append(indicatorvalue_home_smooth,np.average(indicatorvalue_home[0:j+1]))

for j in range(len(indicatorvalue_away)):
    if j > smooth_value:
        indicatorvalue_away_smooth=np.append(indicatorvalue_away_smooth,np.average(indicatorvalue_away[j-smooth_value:j]))
    else:
        indicatorvalue_away_smooth=np.append(indicatorvalue_away_smooth,np.average(indicatorvalue_away[0:j+1]))

plt.plot(frame_passes_home[:,0]/60/5,indicatorvalue_home_smooth,'b',label='Home team')
plt.plot(frame_passes_away[:,0]/60/5,indicatorvalue_away_smooth,'r',label='Away team')

plt.xlabel('Minutes in the match')
plt.title('Indicator value')
plt.legend()


plt.figure()
plt.grid()

from scipy import integrate
#Integrated metric
integrated_indicatorvalue_home = np.array([])
for i in range(len(indicatorvalue_home)):
    integrated_indicatorvalue_home = np.append(integrated_indicatorvalue_home,integrate.simpson(indicatorvalue_home[i:])/(len(indicatorvalue_home)-i))

integrated_indicatorvalue_away = np.array([])
for i in range(len(indicatorvalue_away)):
    integrated_indicatorvalue_away = np.append(integrated_indicatorvalue_away,integrate.simpson(indicatorvalue_away[i:])/(len(indicatorvalue_away)-i))


for goal in frame_goals:
    if int(goal[1]) == home_id:
        plt.axvline(x=goal[0]/60/5,color='b', label ='Home team scores')
    else:
        plt.axvline(x=goal[0]/60/5,color='r', label ='Away team scores')


plt.plot(frame_passes_home[:,0]/60/5,integrated_indicatorvalue_home,'b',label='Home team')
plt.plot(frame_passes_away[:,0]/60/5,integrated_indicatorvalue_away,'r',label='Away team')

plt.xlabel('Minutes in the match')
plt.title('Integrated indicator value')
plt.legend()
                                               
plt.show()






