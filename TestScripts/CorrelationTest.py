import numpy as np
from scipy.signal import butter, sosfilt # Functions for applying a lowpass butterworth
from scipy import signal


def butter_filter(data):
    # Calls function to find the filter coefficents
    # sos = [[ 2.13961520e-05,  4.27923041e-05,  2.13961520e-05,  1.00000000e+00, -1.44797260e+00,  7.75679511e-01],
    #        [ 1.00000000e+00,  2.00000000e+00,  1.00000000e+00, 1.00000000e+00, -1.37919243e+00,  7.98024930e-01],
    #        [ 1.00000000e+00,  0.00000000e+00, -1.00000000e+00,  1.00000000e+00, -1.56677322e+00,  8.33319209e-01],
    #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.40423528e+00,  9.14112629e-01],
    #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.69358308e+00,  9.37816513e-01]]
    
    sos = [[ 0.31761272,  0.63522543,  0.31761272,  1.,          1.08905692,  0.35395093],
           [ 1.,          2.,          1.,          1.,          1.37178046,  0.69698039],
           [ 1.,          0.,         -1.,          1.,         -0.4360535,  -0.47056428],
           [ 1.,         -2.,          1.,          1.,         -1.89890679,  0.90276295],
           [ 1.,         -2.,          1.,          1.,         -1.95866335,  0.96255059]]
    # Uses filter coefs to filter the data
    filtered = sosfilt(sos, data)
    return filtered


filelog = r'../Recorded Data 2/20240422_124451.csv'
print(filelog)
realData = np.loadtxt(filelog, delimiter=',')
print("test")
input1 = butter_filter(realData[:,0])
input2 = butter_filter(realData[:,3])
input3 = butter_filter(realData[:,4])
input4 = butter_filter(realData[:,6])
# input1 = realData[:,0]
# input2 = realData[:,3]
# input3 = realData[:,4]
# input4 = realData[:,6]
print("test")
counter = 0
jcounter = 0
Datarate = 10000
peak_index1 = np.zeros((np.size(input1)-100))
peak_index2 = np.zeros((np.size(input1)-100))
peak_index3 = np.zeros((np.size(input1)-100))
peak_index4 = np.zeros((np.size(input1)-100))
timeDiff = []
print("test")
for x in range((np.size(input1)-100)):
    counter = counter+1
    peak_index1[x] = (np.argmax(signal.correlate(input1[counter:counter+100],input2[counter:counter+100])))-(100-1)
    peak_index2[x] = (np.argmax(signal.correlate(input2[counter:counter+100],input3[counter:counter+100])))-(100-1)
    peak_index3[x] = (np.argmax(signal.correlate(input3[counter:counter+100],input4[counter:counter+100])))-(100-1)
    #peak_index4[x] = (np.argmax(signal.correlate(input4[counter:counter+100],input1[counter:counter+100])))-(100-1)
    if peak_index1[x] != 0 and peak_index2[x] != 0 and peak_index3[x] != 0: # and peak_index4[x] != 0:
        jcounter = jcounter+1
        timeDiff.append([peak_index1[x], peak_index2[x], peak_index3[x]]) #, peak_index4[x]])
print("test")
print([row[0] for row in timeDiff])
avetimeDiff = [np.mean([row[0] for row in timeDiff])/Datarate, np.mean([row[1] for row in timeDiff])/Datarate, np.mean([row[2] for row in timeDiff])/Datarate] #, abs(np.mean([row[3] for row in timeDiff])/Datarate)]

print(avetimeDiff)

# initial_guess = [25, 25, 1]

# zed = 1
# array_size = 50
# spacing = 0.125
# middle = array_size/2
#position1 = np.array([middle-((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed])
#position2 = np.array([middle-((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed])
#position3 = np.array([middle-((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed])
#position4 = np.array([middle-((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed])
#position5 = np.array([middle,middle+(4*spacing),zed])
#position6 = np.array([middle,middle+(3*spacing),zed])
#position7 = np.array([middle,middle+(2*spacing),zed])
#position8 = np.array([middle,middle+spacing,zed])
#position9 = np.array([middle+((spacing)*np.sin(1.0472)),middle-((spacing)*np.cos(1.0472)),zed])
#position10 = np.array([middle+((spacing*2)*np.sin(1.0472)),middle-((spacing*2)*np.cos(1.0472)),zed])
#position11 = np.array([middle+((spacing*3)*np.sin(1.0472)),middle-((spacing*3)*np.cos(1.0472)),zed])
#position12 = np.array([middle+((spacing*4)*np.sin(1.0472)),middle-((spacing*4)*np.cos(1.0472)),zed])

# mic1 = [24.44, .952, 1]
# mic5 = [24.813, 1.3269, 1]
# mic7 = [25, 1.8415, 1]
# mic9 = [25, 2.1209, 1]


