from fbpy.gamepack import GamePackToken, TeamType, TrackingFrameRate
from fbpy.opta.enumerations import OptaEventTypes

from scipy.stats import norm
import scipy.stats as stats
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from fbpy.gamepack import GamePack
from fbpy.pitchcontrol import MatchProcessor
import fbpy as fbpy
from mplsoccer import Pitch, VerticalPitch

# Importing match data:
source_directory = "data" # if databricks, use "/dbfs/mnt/gamepacks"
season_id = "2020"  # can be integer too, function will convert to str
competition = "PremierLeague"  # the exact name in the competition folder
match_id = 2128362  # ID of match, the name of the gamepack directory
fps5 = TrackingFrameRate.FPS5  # selects which subdir to load (25fps or 5fps)

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




# Recording who is a goalkeeper:
goalkeeper_ids = []
for player in fbmatch.metadata.home_roster:
    if player.is_goalkeeper():
        goalkeeper_ids.append(player.player_id)
for player in fbmatch.metadata.away_roster:
    if player.is_goalkeeper():
        goalkeeper_ids.append(player.player_id)


# Iterate through all frames in match

frame_goals = np.empty(shape=[0,2]) # First column is the frame of the goal
                                      # second column is the id of the team which scored
frame_passes = np.empty(shape=[0,3]) # First column is the frame of the goal
                                       # second column is the id of the team which scored
                                       # third column is the length of the pass
xpos_home = np.array([])
xpos_away = np.array([])

TIME=np.array([])
nframes=0

for frame in fbmatch:
    nframes+=1
    hometeam = frame.hometeam
    awayteam = frame.awayteam
    for event in frame.events:
        if event.type_id == OptaEventTypes.Goal:
            frame_goals = np.vstack((frame_goals, [int(frame.frame_id),int(event.team_id)]))

    if not frame:
        continue

    else:
        xpos_frame_home=0
        for player in hometeam:
            if player.player_id in goalkeeper_ids:
                continue
            xpos_frame_home+= player.x_pos

        xpos_frame_away=0   
        for player in awayteam:
            if player.player_id in goalkeeper_ids:
                continue
            xpos_frame_away+= player.x_pos

        TIME=np.append(TIME,nframes/60/5)
        xpos_home=np.append(xpos_home,np.average(xpos_frame_home))
        xpos_away=np.append(xpos_away,np.average(xpos_frame_away))



frame_goals=frame_goals.astype(int)

xpos_home = xpos_home/10/100 # Adjusting for units and number of players
xpos_away = -1*xpos_away/10/100 #Multiplied by -1 to account for inversed pitch position relative to home position


min_smooth=1
smooth_value = int(min_smooth*60*5)

xpos_home_smooth = np.array([])
for j in range(len(xpos_home)):
    if j > smooth_value:
        xpos_home_smooth=np.append(xpos_home_smooth,np.average(xpos_home[j-smooth_value:j]))
    else:
        xpos_home_smooth=np.append(xpos_home_smooth,np.average(xpos_home[0:j+1]))


xpos_away_smooth = np.array([])
for j in range(len(xpos_away)):
    if j > smooth_value:
        xpos_away_smooth=np.append(xpos_away_smooth,np.average(xpos_away[j-smooth_value:j]))
    else:
        xpos_away_smooth=np.append(xpos_away_smooth,np.average(xpos_away[0:j+1]))

print('With respect to their own goal:')


print("Average home pitch position before the goal: %f (m)" % (np.average(xpos_home[:frame_goals[0,0]])))
print("Average home pitch position after the goal: %f (m)" % (np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]])))

print("Average away pitch position before the goal: %f (m)" % (np.average(xpos_away[:frame_goals[0,0]])))
print("Average away pitch position after the goal: %f (m)" % (np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]])))



#Supposing the pitch comprises 4 standard deviations

xpos_home_normalised = xpos_home/52.5*4
xpos_away_normalised = xpos_away/52.5*4

indicatorvalue_home_before = stats.norm.cdf(np.average(xpos_home_normalised[:frame_goals[0,0]]))
indicatorvalue_home_after = stats.norm.cdf(np.average(xpos_home_normalised[frame_goals[0,0]:frame_goals[1,0]]))


indicatorvalue_away_before = stats.norm.cdf(np.average(xpos_away_normalised[:frame_goals[0,0]])) 
indicatorvalue_away_after = stats.norm.cdf(np.average(xpos_away_normalised[frame_goals[0,0]:frame_goals[1,0]]))

print('Offensiveness home value (from 0 to 1): ')
print("Before the goal: %f "%(indicatorvalue_home_before))
print("After the goal: %f "%(indicatorvalue_home_after))

print('Offensiveness away value (from 0 to 1): ')
print("Before the goal: %f "%(indicatorvalue_away_before))
print("After the goal: %f "%(indicatorvalue_away_after))

plt.figure()
plt.grid()
plt.ylabel('Averaged x (m)')
plt.xlabel('time (minutes)')
plt.title('Team position before and after conceding a goal (smooth)')
plt.plot(TIME,xpos_home_smooth, 'b', label='Home team')
plt.plot(TIME,xpos_away_smooth, 'r', label='Away team')
for goal in frame_goals:
    if int(goal[1]) == home_id:
        plt.axvline(x=goal[0]/60/5,color='purple', label ='Home goal')
    else:
        plt.axvline(x=goal[0]/60/5,color='green', label ='Away goal')

plt.legend()

##fit_before,V_before=np.polyfit(TIME_before,xpos_before, 1, cov=True)
##fit_after,V_after=np.polyfit(TIME_after,xpos_after, 1, cov=True)
##plt.plot(TIME_before,TIME_before*fit_before[0]+fit_before[1],'r')
##plt.plot(TIME_after,TIME_after*fit_after[0]+fit_after[1],'r')




plt.figure()
plt.grid()
mu = 0
sigma = 52.5/4
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
plt.xlim([mu - 4*sigma, mu + 4*sigma])
plt.ylim([0,np.amax(stats.norm.pdf(x, mu, sigma))*1.1])
plt.xlabel('Averaged x (m)')
plt.title('Distribution of the position average as a gaussian function')
x_home_before = np.linspace(mu - 4*sigma, np.average(xpos_home[:frame_goals[0,0]]), 1000)
x_home_after = np.linspace(mu - 4*sigma, np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), 1000)

x_away_before = np.linspace(mu - 4*sigma, np.average(xpos_away[:frame_goals[0,0]]), 1000)
x_away_after = np.linspace(mu - 4*sigma, np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), 1000)

plt.fill_between(x_home_before, stats.norm.pdf(x_home_before , mu, sigma),0, alpha=0.3, color='purple')
plt.fill_between(x_home_after, stats.norm.pdf(x_home_after, mu, sigma),0, alpha=0.2, color='brown')

plt.fill_between(x_away_before, stats.norm.pdf(x_away_before , mu, sigma),0, alpha=0.3, color='orange')
plt.fill_between(x_away_after, stats.norm.pdf(x_away_after, mu, sigma),0, alpha=0.2, color='green')


plt.plot(x, stats.norm.pdf(x, mu, sigma))
if frame_goals[0,1] == home_id:
    plt.plot(np.average(xpos_home[:frame_goals[0,0]]), stats.norm.pdf(np.average(xpos_home[:frame_goals[0,0]]), mu, sigma),
             color='purple', marker = 'o', markersize = '12', label='Home before scoring')
    plt.plot(np.average(xpos_away[:frame_goals[0,0]]), stats.norm.pdf(np.average(xpos_away[:frame_goals[0,0]]), mu, sigma),
             color='brown', marker = 'o', markersize = '12', label='Away before conceding')
    plt.plot(np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), stats.norm.pdf(np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), mu, sigma),
             color='green', marker = 'o', markersize = '12', label='Home after scoring')
    plt.plot(np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), stats.norm.pdf(np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), mu, sigma),
             color='orange', marker  = 'o', markersize = '12', label='Away after conceding')
    plt.legend()
elif frame_goals[0,1] == away_id:
    plt.plot(np.average(xpos_home[:frame_goals[0,0]]), stats.norm.pdf(np.average(xpos_home[:frame_goals[0,0]]), mu, sigma),
             color='purple', marker = 'o', markersize = '12', label='Home before conceding')
    plt.plot(np.average(xpos_away[:frame_goals[0,0]]), stats.norm.pdf(np.average(xpos_away[:frame_goals[0,0]]), mu, sigma),
             color='brown', marker = 'o', markersize = '12', label='Away before scoring')
    plt.plot(np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), stats.norm.pdf(np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), mu, sigma),
             color='green', marker = 'o', markersize = '12', label='Home after conceding')
    plt.plot(np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), stats.norm.pdf(np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), mu, sigma),
             color='orange', marker  = 'o', markersize = '12', label='Away after scoring')
    plt.legend()

# CONTINUOUS INDICATOR VALUE
indicatorvalue_home = np.array([])
for value in xpos_home_normalised:
    indicatorvalue_home = np.append(indicatorvalue_home, norm.cdf(value))
    
indicatorvalue_away = np.array([])
for value in xpos_away_normalised:
    indicatorvalue_away = np.append(indicatorvalue_away, norm.cdf(value))


plt.figure()



plt.grid()
for goal in frame_goals:
    if int(goal[1]) == home_id:
        plt.axvline(x=goal[0]/60/5,color='purple', label ='Home goal')
    else:
        plt.axvline(x=goal[0]/60/5,color='green', label ='Away goal')
        
min_smooth=1
smooth_value = int(min_smooth*60*5)
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

plt.plot(TIME,indicatorvalue_home_smooth,'b',label='Home team')
plt.plot(TIME,indicatorvalue_away_smooth,'r',label='Away team')

plt.xlabel('Minutes in the match')
plt.title('Average x position indicator value')
plt.legend()

plt.figure()
pitch = Pitch(pitch_type='tracab',  # example plotting a tracab pitch
              pitch_length=105, pitch_width=68,
              axis=True, label=True)
pitch.draw()

if frame_goals[0,1] == home_id:
    plt.axvline(x=100*np.average(xpos_home[:frame_goals[0,0]]), color='lime', label='Home before scoring')
    plt.axvline(x=-100*np.average(xpos_away[:frame_goals[0,0]]), color='peru', label='Away before conceding')
    plt.axvline(x=100*np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), color='green', label='Home after scoring')
    plt.axvline(x=-100*np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), color='brown', label='Away after conceding')
    
elif frame_goals[0,1] == away_id:
    plt.axvline(x=100*np.average(xpos_home[:frame_goals[0,0]]), color='lime', label='Home before conceding')
    plt.axvline(x=-100*np.average(xpos_away[:frame_goals[0,0]]), color='peru', label='Away before scoring')
    plt.axvline(x=100*np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]]), color='green', label='Home after conceding')
    plt.axvline(x=-100*np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]]), color='brown', label='Away after scoring')
plt.legend()
plt.fill_between([100*np.average(xpos_home[:frame_goals[0,0]]), 100*np.average(xpos_home[frame_goals[0,0]:frame_goals[1,0]])],[-6800,-6800], [6800, 6800], alpha=0.3, color='green')
plt.fill_between([-100*np.average(xpos_away[:frame_goals[0,0]]), -100*np.average(xpos_away[frame_goals[0,0]:frame_goals[1,0]])],[-6800,-6800],[6800, 6800], alpha=0.3, color='brown')

plt.show()
