__author__ = 'Copyright (c) 2017 Minggui Chen All rights reserved.'
"""
PyTimingTask:
Package implementing the two-interval timing task in mice, using Arduino
Fri Apr 07 2017 by Minggui Chen

Note: 1) pip install pymata==2.14, and replace installed PyMata with the modified files
      2) upload modified firmataplus in folder "Arduino ino" to arduino, using 
         the method describe in the pdf file
      3) when the lickometer malfunctions for unknown reasons, upload Lickometer.ino
         first, and then upload modified firmataplus again
Usage: 1) set all the experimental parameters early in the file
       2) ctrl+c to quit the ongoing experiment, with data saved automatically
       3) trial-by-trial visualization available for the experiementer
       4) all based on arduino mega 2560, with ~1000 Hz sampling rate
"""

import signal, time, inspect, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from PyMata.pymata import PyMata
import random
import tkinter as tk
from multiprocessing.pool import ThreadPool

warnings.filterwarnings("ignore")

# Define all experimental parameters here
animalTag = 'm209_080221_opto_18'                                           #set the filename of your data here
numTrials = 400                                               #number of trials
countErrorTrial = True                                         #whether or not count error trials
servoTime = 1000                                                 #wait for servo completion, in ms
numEpochs = 2                                                   #number of experimental epochs
toneFrequency = 10000                                            #frequencies of the consective tones
toneDuration = 150
numIntervals = 2                                                #number of target intervals
intervals = [[500, 800, 1100], [2100, 1800, 1500]]                                    #exact intervals to be tested, in ms
n_intervals = 1
lickWindow = 5000                                       #time window for reward, in ms
breakLickEarly = True                                           #whether or not quit the trial when licking early
rewardDuration = 300                                          #how long you want to deliver reward
vacuumDuration = 50
airpuffDuration = 150                                            # airpuff punishment
timeout = 3000                                                    #timeout for incorrect trials
iti = 8000                                                      #inter trial interval, in ms
Punishment = False
Activate_opto = False
percentage_opto_trials = 0.35

filesep = '/'
pyPath = os.getcwd()
pyPath = (pyPath, animalTag + "_Data.npz")

dataPath = filesep.join(pyPath)                                 #path to save the data, relative to the cwd
timeStamp = []                                                  #hardware timestamps from Arduino in each trial                                       
lickStatus_1 = []                                                 #recorded lickometer status in each trial
lickStatus_2 = []
trialStartStamp = []                                            #start timestamps in each trial
toneStamp = []                                                  #timestamps of tones in each trial
targetInterval = []                                             #target timing to be probed in each trial
responseType = []                                               #response type: True - correct; False - error
responseTime = []                                               #reaction time in each trial
IntervalType = []
Strategy = []
lickclock_1 = []
lickclock_2 = []
iti_list = []
presentation = []
handletouchTime = []
timeTrigger = []
ledTime = []
trialStamp = []
opto_trials = []

# Define all hardware configs here
arduinoPort = 'COM3'                                            #port ID of arduino
servoPin = 3                                                    #pin ID for servo control
#servoInitiationPin = 4
lickometerPin_1 = 30
lickometerPin_2 = 24
lickometerRegister = 0                                          #data register of elec0 touch event
triggerPin = 22                                                 #pin ID for trigger output
speakerPin = 2                                                  #speaker pin, must be digital and PWM 
rewardPin_1 = 48                                                #pin used to control the gate for the reward
rewardPin_2 = 44
vacuumPin = 40
airpuffPin = 42
BOARD_LED = 10
#initiationPin = 34
CLOCKPIN = 0                                                    #CONSTANT pin for the hardware clock from Arduino
Opto_inhibition_pin_1 = 35
Opto_inhibition_pin_2 = 37

# Global state variables to control the data flow
enableRecording = False                                         #enable/disable recording
enableTrial = False                                             #whether it's within a trial
enableLick_1 = False      
enableLick_2 = False                                       #whether it's within the allowable lick time window
Normal = True
Forced_Right = False  # 0
Forced_Left = False     # 1
OPTO_ON = False

bLicked_1 = []
bLicked_2 = []

#response types
class response:
    NONE = 0
    CORRECT = 1
    INCORRECT = 2

# Configure arduino using PyMata
# Report all the data and timestamps at almost 1 KHz, no distortion
#   set sampling interval as 19 ms in the Arduino and 1 ms in pymata
# Sampling rate will decrease as a function of AO channels
#   1-4 channels: 1 KHz; Plus I2C: ~800 Hz
#initiate Aruino at a baud rate of 115200 and sampling rate of 1000 Hz
board = PyMata(arduinoPort, False, False)                        
board.set_sampling_interval(1)
#signal handler for user abort with Ctrl+C
def SignalHandler(sig, frame):
    print('User aborted. Cleaning up...')
    SaveData()
    if board is not None:
        board.reset()
    print('Data saved. Done.')
signal.signal(signal.SIGINT, SignalHandler)

#enable/disable recording
def EnableRecording(status):
    global enableRecording
    enableRecording = status
EnableRecording(False)

#spout control
board.servo_config(servoPin)
def EnableSpout(status):
    global servoPin
    if (status):  #spout on, servo to degs
        board.analog_write(servoPin, 120)
    else:       #spout off, servo to degs
        board.analog_write(servoPin, 40)
    time.sleep(1000/1000)

#reward control
board.set_pin_mode(rewardPin_1, board.OUTPUT, board.DIGITAL)
board.set_pin_mode(rewardPin_2, board.OUTPUT, board.DIGITAL)

def Reward(duration, rewardPin):
    for i in range(np.int(duration/100)):
        board.digital_write(rewardPin, 1)
        time.sleep(80/1000.0)
        board.digital_write(rewardPin, 0)
        time.sleep(80/1000.0)

#vaccuum control
board.set_pin_mode(vacuumPin, board.OUTPUT, board.DIGITAL)
def Vacuum(duration):
    global vacuumPin
    board.digital_write(vacuumPin, 1)
    time.sleep(duration/1000.0)
    board.digital_write(vacuumPin, 0)

#airpuff control
board.set_pin_mode(airpuffPin, board.OUTPUT, board.DIGITAL)
def Airpuff(duration):
    global airpuffPin, Punishment
    if Punishment == True:
        board.digital_write(airpuffPin, 1)
        time.sleep(duration/1000.0)
        board.digital_write(airpuffPin, 0)

#LED for go cue
def PlayLED():
    global BOARD_LED
    board.set_pin_mode(BOARD_LED, board.OUTPUT, board.DIGITAL)
    board.digital_write(BOARD_LED, 1)
    time.sleep(0.001)
    board.digital_write(BOARD_LED, 0)       

board.set_pin_mode(Opto_inhibition_pin_1, board.OUTPUT, board.DIGITAL)
board.set_pin_mode(Opto_inhibition_pin_2, board.OUTPUT, board.DIGITAL)
#
def opto_inhibition(opto_pin, duration):
    board.digital_write(opto_pin, 1)
    time.sleep(duration/1000)
    board.digital_write(opto_pin, 0)

#lickometer input
def LickometerCallback_1(data_1):
    global thisResponseType, thisResponseTime, enableRecording, enableTrial, enableLick_1,\
        thisInterval, thisLickStatus_1, breakLickEarly, lickWinClock, thisLickClock_1, rewardDuration, reward, thistrialStamp, bLicked_1
    bLicked_1.append(data_1[2])
    if enableRecording:
        thisLickStatus_1.append(bLicked_1)
    if enableTrial and not thisResponseType:
        if (enableLick_1 and len(bLicked_1)>1):                                            #lick correctly
            thisResponseType = response.CORRECT
            Reward(rewardDuration, reward)
            thistrialStamp.append(clock)
            thisLickClock_1.append(clock)
            print('touch 1')
        elif (breakLickEarly and clock>lickWinClock and not enableLick_1 and len(bLicked_1) >5):                   #lick early
            thisResponseType = response.INCORRECT
            Airpuff(airpuffDuration)
            thisLickClock_1.append(clock)
            thistrialStamp.append(clock)
            EnableSpout(False)
            print('touch 1 *')
        if thisResponseType and bLicked_1:
            thisResponseTime = clock

    board.set_digital_latch(lickometerPin_1, board.DIGITAL_LATCH_HIGH, LickometerCallback_1)
board.set_pin_mode(lickometerPin_1, board.INPUT, board.DIGITAL)
board.set_digital_latch(lickometerPin_1, board.DIGITAL_LATCH_HIGH, LickometerCallback_1)


#lickometer input
def LickometerCallback_2(data_2):
    global thisResponseType, thisResponseTime, enableRecording, enableTrial, enableLick_2, \
        thisInterval, thisLickStatus_2, breakLickEarly, lickWinClock, thisLickClock_2, rewardDuration, reward, thistrialStamp, bLicked_2
    bLicked_2.append(data_2[2])
    # print(len(bLicked_2))
    if enableRecording:
        thisLickStatus_2.append(bLicked_2)
    if enableTrial and not thisResponseType:
        if (enableLick_2 and len(bLicked_2)>1):                                            #lick correctly
            thisResponseType = response.CORRECT
            Reward(rewardDuration, reward)
            thisLickClock_2.append(clock)
            thistrialStamp.append(clock)
            print('touch 2')
        elif (breakLickEarly and clock>lickWinClock and not enableLick_2 and len(bLicked_2) >5):                   #lick early
            thisResponseType = response.INCORRECT
            Airpuff(airpuffDuration)
            thisLickClock_2.append(clock)
            thistrialStamp.append(clock)
            EnableSpout(False)
            print('touch 2 *')
        if thisResponseType and bLicked_2:
            thisResponseTime = clock
    board.set_digital_latch(lickometerPin_2, board.DIGITAL_LATCH_HIGH, LickometerCallback_2)
board.set_pin_mode(lickometerPin_2, board.INPUT, board.DIGITAL)
board.set_digital_latch(lickometerPin_2, board.DIGITAL_LATCH_HIGH, LickometerCallback_2)

#external trigger
board.set_pin_mode(triggerPin, board.OUTPUT, board.DIGITAL)
def SendTrigger(duration):
    global triggerPin
    board.digital_write(triggerPin, 1)
    time.sleep(duration/1000)
    board.digital_write(triggerPin, 0)
#tone playing in non-blocked fashion
def PlayTone(frequency, duration):
    global speakerPin
    board.play_tone(speakerPin, board.TONE_TONE, frequency, duration)
 
    
#timestamps in ms from Arduino hardware, much more accurate 
clock = np.float64(0.0)
threshold = 500
numClockReset = 0.0
maxTrialClock = np.nan
lickWinClock = np.nan


def StampCallback(data):
    stamp = data[2]
    global numClockReset, threshold, clock, thisTimeStamp, maxTrialClock, lickWinClock, enableTrial, enableLick_1, enableLick_2, enableRecording, timingID
    if (clock>0 and clock>=threshold+numClockReset*1000 and stamp<threshold):
        numClockReset = numClockReset +1
    clock = stamp +numClockReset*1000
    if not np.isnan(lickWinClock) and clock >= lickWinClock:
        if timingID == 0:
            enableLick_1 = True
            enableLick_2 = False
        if timingID == 1:
            enableLick_2 = True
            enableLick_1 = False
    if not np.isnan(maxTrialClock) and clock>maxTrialClock:  #prevent from over-queued inputs from Arduino
        enableTrial = False
        enableLick_2 = False
        enableLick_2 = False
        enableRecording = False
    if enableRecording:
        thisTimeStamp.append(clock)
    board.set_analog_latch(CLOCKPIN, board.ANALOG_LATCH_GTE, 0, StampCallback)
board.set_pin_mode(CLOCKPIN, board.INPUT, board.ANALOG)
board.set_analog_latch(CLOCKPIN, board.ANALOG_LATCH_GTE, 0, StampCallback)

#wait the system to be stablized
def WaitSystem(duration):
    start = time.time()
    while (time.time()-start <= duration/1000):
        time.sleep(.001)

#clean up the recording system
def Cleanup():
    #EnableLickometer(False)
    board.reset()

def sensory_button():
    global iti, rewardPin_1, rewardPin_2, Normal, Forced_Right, Forced_Left, servoTime, Activate_opto, n_intervals
    print("start")
    def print_value():
        global iti
        value = delay.get()
        iti = int(value)
        print('Change iti to : ' + str(iti))

    def update_servo():
        global servoTime
        servoTime = int(servo_time.get())
        print('Delay of stimulus presentation changed to = ' + str(servoTime) + 'ms')

    def number_intervals():
        global n_intervals
        value = n_inter.get()
        n_intervals = int(value)
        print('Number of Interval Types changed to = ' + str(n_intervals))


    def Reward_1(rewardPin=rewardPin_1):
        for i in range(np.int(400/100)):
            board.digital_write(rewardPin, 1)
            time.sleep(50/1000.0)
            board.digital_write(rewardPin, 0)
            time.sleep(50/1000.0)
        print('right valve')

    def Reward_2(rewardPin=rewardPin_2):
        for i in range(np.int(400/100)):
            board.digital_write(rewardPin, 1)
            time.sleep(40/1000.0)
            board.digital_write(rewardPin, 0)
            time.sleep(40/1000.0)
        print('left valve')

    def f_Normal():
        global Normal, Forced_Right, Forced_Left
        Normal = True
        Forced_Right = False
        Forced_Left = False
        display = tk.StringVar(root, value='NORMAL')
        currect_strategy = tk.Entry(root, width=20, textvariable=display)
        currect_strategy.place(x=150, y=300)
        print('Strategy changes to Normal Alternation')

    def f_ForceR():
        global Normal, Forced_Right, Forced_Left
        Normal = False
        Forced_Right = True
        Forced_Left = False
        display = tk.StringVar(root, value='FORCE RIGHT')
        currect_strategy = tk.Entry(root, width=20, textvariable=display)
        currect_strategy.place(x=150, y=300)
        print('Strategy changes to Force lick to Right')

    def f_ForceL():
        global Normal, Forced_Right, Forced_Left
        Normal = False
        Forced_Right = False
        Forced_Left = True
        display = tk.StringVar(root, value='FORCE LEFT')
        currect_strategy = tk.Entry(root, width=20, textvariable=display)
        currect_strategy.place(x=150, y=300)
        print('Strategy changes to Force lick to Left')

    def Opto():
        global Activate_opto
        if Opto_buttom['background'] == 'red':
            Opto_buttom.configure(background='green')
            Activate_opto = True
            print('Optogenetic activation mode = ON')
        else:
            Opto_buttom.configure(background='red')
            Activate_opto = False
            print('Optogenetic activation mode = OFF')


    w = 320
    h = 400
    x = 50
    y = 100

    root = tk.Tk()
    root.geometry("%dx%d+%d+%d" % (w, h, x, y))
    root.title('Parameter Timing Task')

    initial_iti = tk.StringVar(root, value=str(iti))
    delay = tk.Spinbox(root, from_=5000, to_=20000, increment=1000, textvariable=initial_iti)
    delay.place(x=60, y=5)

    iti_label = tk.Label(root, text="ITI : ")
    iti_label.place(x=10, y=5)

    values_button = tk.Button(root, text='SUMMIT', width=10, command=print_value)
    values_button.place(x=220, y=3)

    servo_label = tk.Label(root, text='Spout\nPresentation\nDelay')
    servo_label.place(x=10, y=35)
    initial_servo_time = tk.StringVar(root, value=str(servoTime))
    servo_time = tk.Entry(root, width=10, textvariable=initial_servo_time)
    servo_time.place(x=100, y=45)
    servo_ms = tk.Label(root, text='ms')
    servo_ms.place(x=160, y=45)
    servo_button = tk.Button(root, text='SUMMIT', width=10, command=update_servo)
    servo_button.place(x=220, y=45)

    initial_n_inter = tk.StringVar(root, value=str(n_intervals))
    n_inter = tk.Spinbox(root, from_=1, to_=3, increment=1, textvariable=initial_n_inter, width=10)
    n_inter.place(x=100, y=100)
    n_inter_label = tk.Label(root, text='N Intervals : ')
    n_inter_label.place(x=10, y=100)
    n_inter_button = tk.Button(root, text='SUMMIT', width=10, command=number_intervals)
    n_inter_button.place(x=220, y=100)

    valve_label = tk.Label(root, text='DELIVER REWARD : ', font='Helvetica 10 bold')
    valve_label.place(x=10, y=130)

    valve_button_right = tk.Button(root, text='Short\nValve', width=10, background='orange', command=Reward_1)
    valve_button_right.place(x=50, y=160)

    valve_button_left = tk.Button(root, text='Long\nValve', width=10, background='dodger blue', command=Reward_2)
    valve_button_left.place(x=190, y=160)

    strategy_label = tk.Label(root, text='CHANGE STRATEGY : ', font='Helvetica 10 bold')
    strategy_label.place(x=10, y=220)

    normal_button = tk.Button(root, text='NORMAL', width=10, background='green yellow', command=f_Normal)
    normal_button.place(x=15, y=250)
    right_button = tk.Button(root, text='FORCE\nSHORT', width=10, background='deep sky blue', command=f_ForceR)
    right_button.place(x=115, y=250)
    left_button = tk.Button(root, text='FORCE\nLONG', width=10, background='medium purple', command=f_ForceL)
    left_button.place(x=215, y=250)

    selected_strategy = tk.Label(root, text='Optogenetic :', font='Helvetica 9 bold')
    selected_strategy.place(x=10, y=300)

    Opto_buttom = tk.Button(root, text='OPTO', width=12, background='red', font= 'Helvetica 10 bold', command=Opto)
    Opto_buttom.place(x=115, y=340)

    root.mainloop()


pool = ThreadPool(processes=1)
async_result = pool.apply_async(sensory_button, ())

# Functions independent of Arduino
#generate the stimulus condition for the next trial 
def GenerateCondition():
    global initial_sequence, responseType, Normal, Forced_Right, Forced_Left, percentage_opto_trials, OPTO_ON, Activate_opto

    if Normal == True:
        ## First select next trial depending on previous 10 trials
        #To keep the even distribution of both conditions
        if np.count_nonzero(initial_sequence==0) > 6:
            next_trial = 1
        elif np.count_nonzero(initial_sequence==1) > 6:
            next_trial = 0
        else:
            next_trial = np.random.choice(np.arange(2), 1)

        if len(responseType)>= 2:
            if responseType[-1] == responseType[-2] == 1 and initial_sequence[-1] == initial_sequence[-2]:
                next_trial = 1 - initial_sequence[-1]
                if random.random() >= 0.9:
                    next_trial = 1 - next_trial

        ## Avoid having more than 3 consecutive trials of one kind.
        if initial_sequence[-1] == initial_sequence[-2]  == initial_sequence[-3] == next_trial:
            next_trial = 1 - next_trial

        ## Avoid alternation
        if initial_sequence[-2] == initial_sequence[-4] == next_trial and initial_sequence[-1] == initial_sequence[-3] == (1 - next_trial):
            next_trial = 1 - next_trial


    if Forced_Right == True:
        ## When fornced to lick to the right (condition 0). Take last 10 trials and try to keep 70% of them to 0.
        if np.count_nonzero(initial_sequence==0) <= 7:
            next_trial = 0
        else:
            next_trial = 1
        #To have a bit of randomness
        if random.random() >= 0.7:
            next_trial = 1 - next_trial

        ## Avoid alternation
        if initial_sequence[-2] == initial_sequence[-4] == next_trial and initial_sequence[-1] == initial_sequence[-3] == (1 - next_trial):
            next_trial = 1 - next_trial

        # if responseType[-1] == responseType[-2] == initial_sequence[-3] == next_trial:
        #     next_trial = 1 - next_trial

        if len(responseType)>= 2:
            if responseType[-1] == responseType[-2] == 1 and initial_sequence[-1] == initial_sequence[-2]:
                next_trial = 1 - initial_sequence[-1]
            if random.random() >= 0.9:
                next_trial = 1 - next_trial
            if initial_sequence[-1] == initial_sequence[-2] == initial_sequence[-3] and responseType[-1] == responseType[-2] == 1:
                next_trial = 1 - initial_sequence[-1]



    if Forced_Left == True:
        ## When fornced to lick to the left (condition 1). Take last 10 trials and try to keep 70% of them to 1.
        if np.count_nonzero(initial_sequence==1) <= 7:
            next_trial = 1
        else:
            next_trial = 0
        #To have a bit of randomness
        if random.random() >= 0.7:
            next_trial = 1 - next_trial

        ## Avoid alternation
        if initial_sequence[-2] == initial_sequence[-4] == next_trial and initial_sequence[-1] == initial_sequence[-3] == (1 - next_trial):
            next_trial = 1 - next_trial

        # if responseType[-1] == responseType[-2] == initial_sequence[-3] == next_trial:
        #     next_trial = 1 - next_trial

        if len(responseType)>= 2:
            if responseType[-1] == responseType[-2] == 1 and initial_sequence[-1] == initial_sequence[-2]:
                next_trial = 1 - initial_sequence[-1]
            if random.random() >= 0.9:
                next_trial = 1 - next_trial
            if initial_sequence[-1] == initial_sequence[-2] == initial_sequence[-3] and responseType[-1] == responseType[-2] == 1:
                next_trial = 1 - initial_sequence[-1]

    if random.random() <= percentage_opto_trials and Activate_opto == True:
        new_responseType = np.asarray(responseType)
        per_correct = (np.count_nonzero(new_responseType[-4:]==1)/4)*100
        print(per_correct)
        if per_correct >=74:
            OPTO_ON = True
            print('Optogenetic inhibition trial')
        else:
            OPTO_ON = False
    else:
        OPTO_ON = False

    return int(next_trial), OPTO_ON

#inter-trial-interval
def ITI(duration):
    time.sleep(duration/1000)
#save the data, either by the end of experiments or after user abortion
def SaveData():
    np.savez(dataPath, timeStamp=timeStamp, lickStatus_1=lickStatus_1, lickStatus_2=lickStatus_2, toneStamp=toneStamp, 
             responseType=responseType, responseTime=responseTime, targetInterval=targetInterval, timelick_1= lickclock_1,
             timelick_2=lickclock_2, iti=iti_list, presentationTime=servoTime, timeTrigger=timeTrigger, ledTime=ledTime,
             trialStamp=trialStamp, opto_trials=opto_trials)


# Stablize the recording system
EnableRecording(False)
EnableSpout(False)
WaitSystem(1000)

# Run trials
#initialize plots
fig = plt.figure(figsize=(5,4))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
plt.show(block=False)
left_spout = 0
right_spout = 0
ax2.set_ylim(0, 100)
ax3.set_xticks([0, 1])
ax3.set_xticklabels(['Right\nCorrect', 'Left\nCorrect'])
Correct_trials = 0
percentage_correct = 0
Noresponse_trials = 0
#trial settings
numTrials = np.ceil(numTrials/numIntervals)*numIntervals
numTrials = int(numTrials)
numRemainedInterval = np.zeros(numIntervals, dtype=np.int) +numTrials/numIntervals
numValidTrials = 0
i = -1
initial_sequence = np.random.choice(np.arange(2), 10)
first_trials = [0, 1, 0, 1, 0]

while numValidTrials < numTrials:
    #number of total trials
    i += 1

    #initialize data buffers for this trial
    WaitSystem(1)
    thisTimeStamp = []
    thisLickStatus_1 = []
    thisLickStatus_2 = []
    thisLickClock_1 = []
    thisLickClock_2 = []
    thisResponseType = response.NONE
    thisResponseTime = np.nan 
    thisToneStamp = []
    led_time = []
    thistrialStamp = []
    maxTrialClock = np.nan
    lickWinClock = np.nan
    enableLick_1 = False
    enableLick_2 = False
    enableTrial = False
    bLicked_1 = []
    bLicked_2 = []

    time.sleep(1)

    if i>=5:
        #generate this stimulus condition return index number
        timingID, OPTO_ON = GenerateCondition()
    else:
        timingID = first_trials[i]


    interval_choice = np.random.choice(np.arange(n_intervals), 1)
    thisInterval = intervals[timingID][interval_choice[0]]
    initial_sequence = np.append(initial_sequence, timingID)
    initial_sequence = initial_sequence[1:]
    print('interval ' + str(thisInterval))

    if i%20 == 0:
        ax1.set_xlim(i-1, i+21)
        ax1.set_yticks(np.arange(1,3))
        ax1.set_yticklabels(['Short', 'Long'], fontsize=12)
        ax1.set_xlabel('trial')
        ax2.set_xlim(i-1, i+21)
    ax1.set_ylim(0.8, 2.2)

    #ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=2)
    if timingID == 0:
        if interval_choice[0] == 0:
            dot_color = 'orange'
            dot_legend = '500ms'
        if interval_choice[0] == 1:
            dot_color = 'darkorange'
            dot_legend = '800ms'
        if interval_choice[0] == 2:
            dot_color = 'orangered'
            dot_legend = '1100ms'
        ax1.plot(i+1, 1, 'o', markersize=10, color=dot_color, alpha=0.8, label=dot_legend)
        resp = 1
    if timingID == 1:
        if interval_choice[0] == 2:
            dot_color = 'lightblue'
            dot_legend = '1500ms'
        if interval_choice[0] == 1:
            dot_color = 'dodgerblue'
            dot_legend = '1800ms'
        if interval_choice[0] == 0:
            dot_color = 'lightskyblue'
            dot_legend = '2100ms'

        ax1.plot(i+1, 2, 'o', markersize=10, color=dot_color, alpha=0.8, label=dot_legend)
        resp = 2
    if OPTO_ON == True:
        ax1.plot(i+1, resp, 'o', markersize=14, fillstyle='none', markeredgecolor='lime', markeredgewidth=4, alpha=0.5)

    fig.canvas.flush_events()
    plt.pause(.001)


    #start the trial
    print("Starting trial %s" % (i+1))
    SendTrigger(5)
    thistrialStamp.append(clock)
    EnableRecording(True)
    
    trialStartStamp.append(clock)
    enableTrial = True


    thisToneStamp.append(clock)
    initial_cue = thisToneStamp[0] + 2000
    thistrialStamp.append(clock)
    while clock <= initial_cue:
        PlayLED()
        time.sleep(.001)


    if timingID == 0:
        enableLick_1 = True
        reward = rewardPin_1
        opto = Opto_inhibition_pin_1
        print('lick right - short - 1')
    if timingID == 1:
        enableLick_2 = True
        reward = rewardPin_2
        opto = Opto_inhibition_pin_2
        print('lick left - long - 2')
    if i<=10:
       Reward(200, reward)
    time.sleep(2.0)

    PlayTone(toneFrequency, toneDuration)
    thistrialStamp.append(clock)
    thisToneStamp.append(clock)
    if OPTO_ON == True:
        opto_inhibition(opto, 5)

    while clock-thisToneStamp[-1] < thisInterval:
        time.sleep(.01)
    
    thisToneStamp.append(clock)
    PlayTone(toneFrequency, toneDuration)
    thistrialStamp.append(clock)
    time.sleep(0.1)

    # if OPTO_ON == True:
    #     opto_inhibition(opto, 5)

    spout_presentation = thisToneStamp[-1] + servoTime + toneDuration

    while clock < spout_presentation:
        time.sleep(.001)

    lickWinClock = spout_presentation + 200
    maxTrialClock = lickWinClock + lickWindow

    #spout delivery
    thisToneStamp.append(clock)
    thistrialStamp.append(clock)
    EnableSpout(True)

    while clock <= maxTrialClock:
        time.sleep(.001)
        if thisResponseTime>=lickWinClock and thisResponseType:
            break
    
    enableTrial = False
    enableLick_1 = False
    enableLick_2 = False


    #stop recording

    if thisResponseType == response.CORRECT:
        print('Rewarded')

        ax1.plot(i+1, resp, 'o', markersize=10, fillstyle='none', markeredgecolor='green', markeredgewidth=1.5)
        if timingID == 0:
            right_spout = right_spout + 1
        if timingID == 1:
            left_spout = left_spout + 1
        ax3.bar([0, 1], [right_spout, left_spout], width=0.5, color=['orangered', 'dodgerblue'])
        ax3.set_ylim(0, max(right_spout, left_spout))
        Correct_trials = Correct_trials + 1

        time.sleep(2.000)
        
    if thisResponseType==response.INCORRECT:
        print('Incorrect')
        ax1.plot(i+1, resp, 'o', markersize=10, fillstyle='none', markeredgecolor='yellow', markeredgewidth=1.5)
        
    if thisResponseType==response.NONE:
        #Reward(rewardDuration, reward)
        print('No response')
        ax1.plot(i+1, resp, 'o', markersize=10, fillstyle='none', markeredgecolor='red', markeredgewidth=1.5)
        Noresponse_trials = Noresponse_trials + 1

        #time.sleep(2.000)

    EnableRecording(False)
    trialStopStamp = clock   
    EnableSpout(False)
    thistrialStamp.append(clock)
    # touch_handle = False

    if Correct_trials != 0:
        percentage_correct = (Correct_trials/(i+1-Noresponse_trials))
        percentage_correct = percentage_correct*100.0

    ax2.plot(i+1, percentage_correct, 'o', markersize=2, color='mediumblue')
    ax2.plot(np.arange(numTrials+1), (np.ones(numTrials+1))*50, '-', markersize=0.3, color='k')
    ax2.plot(np.arange(numTrials+1), (np.ones(numTrials+1))*75, '-', markersize=0.2, color='g')
    fig.canvas.flush_events()
    plt.pause(.001)

    # Vacuum(vacuumDuration)

    #truncate the data
    thisTimeStamp = np.asarray(thisTimeStamp) - trialStartStamp[0]
    thisLickStatus_1 = np.asarray(thisLickStatus_1)
    thisLickStatus_2 = np.asarray(thisLickStatus_2)
    tonesStamp = [thisToneStamp[2], thisToneStamp[3]]
    led_time = [thisToneStamp[0], thisToneStamp[1]]


    #transfer buffered data in this trial to the output
    timeTrigger.append(trialStartStamp)
    timeStamp.append(thisTimeStamp)
    lickStatus_1.append(thisLickStatus_1)
    lickStatus_2.append(thisLickStatus_2)
    toneStamp.append(tonesStamp)
    ledTime.append(led_time)
    responseType.append(thisResponseType)
    responseTime.append(thisResponseTime)
    targetInterval.append(thisInterval)
    lickclock_1.append(thisLickClock_1)
    lickclock_2.append(thisLickClock_2)
    iti_list.append(iti)
    presentation.append(servoTime)
    trialStamp.append(thistrialStamp)
    opto_trials.append(OPTO_ON)


    #update valid trial numbers
    if (not countErrorTrial and thisResponseType==response.CORRECT) or countErrorTrial:
        numValidTrials += 1
        numRemainedInterval[timingID] -= 1
        
    #inter-trial-interval
    while clock-trialStopStamp <= iti:
        time.sleep(.001)
    
    #timeout for incorrect trials
    if thisResponseType == response.INCORRECT:
        WaitSystem(timeout)

    #plots for the results
    #  update the background canvas every 20 trials
    if i%20 == 0:
        plt.xlim()
        plt.ylim((i+.5, i+20.5))

# Save the results
SaveData()

# Stop recording and clean up
Cleanup()


