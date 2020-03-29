# This vibrato detection algorithm works for notes and tested on MTG Violin recordings.
# It does not work for Carlos's transverse flute non vibrato recordings. Noise elimination is needed.

import pandas as pd
from derivative_parameter_calculator import derivative_parameter_calculator

# First get peak percentage, mean value, max value, max location

# file = "Neumman-04 render 005.wav"
df = pd.read_csv('violin_vibrato.csv')
files = []

# Selecting which catagory to be anaylzed
for i in range(len(df.File_Name)):
    if df.Presence_of_Vibrato[i] == 'no vibrato':
        files.append(df.File_Name[i])

for file in files:

    percentage,mean,max,max_loc = derivative_parameter_calculator(file)

    print file,"Peak percentage: ",percentage,"Mean value: ",mean, "Max value:",max, "Location of max value (%): ",max_loc

    # Then, see if vibrato might be present
    # Check peak percentage rate
    if percentage > 5:
        vibrato = True
    elif percentage < 3:
        vibrato = False
    elif percentage < 5 and percentage > 3:
        if max < 150:
            vibrato = True
        elif max_loc < 0.3:
            vibrato = True
        else:
            vibrato = False

    if vibrato:
        print "Vibrato is present in the sound file, ", file
    else:
        print "Vibrato is not present in the sound file, ", file

