# import numpy as np
# from scipy.signal import butter, sosfilt # Functions for applying a lowpass butterworth

# def cross_correlation(mic1, mic2):
#     # Reverse the second signal
#     mic2_flipped = np.flip(mic2)
#     # Calculate the cross-correlation
#     return np.correlate(mic1, mic2_flipped, mode='full')

# def butter_filter(data):
#     # Calls function to find the filter coefficents
#     # sos = [[ 2.13961520e-05,  4.27923041e-05,  2.13961520e-05,  1.00000000e+00, -1.44797260e+00,  7.75679511e-01],
#     #        [ 1.00000000e+00,  2.00000000e+00,  1.00000000e+00, 1.00000000e+00, -1.37919243e+00,  7.98024930e-01],
#     #        [ 1.00000000e+00,  0.00000000e+00, -1.00000000e+00,  1.00000000e+00, -1.56677322e+00,  8.33319209e-01],
#     #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.40423528e+00,  9.14112629e-01],
#     #        [ 1.00000000e+00, -2.00000000e+00,  1.00000000e+00,  1.00000000e+00, -1.69358308e+00,  9.37816513e-01]]
    
#     sos = [[ 0.31761272,  0.63522543,  0.31761272,  1.,          1.08905692,  0.35395093],
#            [ 1.,          2.,          1.,          1.,          1.37178046,  0.69698039],
#            [ 1.,          0.,         -1.,          1.,         -0.4360535,  -0.47056428],
#            [ 1.,         -2.,          1.,          1.,         -1.89890679,  0.90276295],
#            [ 1.,         -2.,          1.,          1.,         -1.95866335,  0.96255059]]
#     # Uses filter coefs to filter the data
#     filtered = sosfilt(sos, data)
#     return filtered

# filelog = r'../Recorded Data 2/20240422_123302.csv'
# print(filelog)
# realData = np.loadtxt(filelog, delimiter=',')

# # input1 = butter_filter(realData[:,0])
# # input2 = butter_filter(realData[:,3])
# # input3 = butter_filter(realData[:,4])
# # input4 = butter_filter(realData[:,6]
# input1 = realData[:,0]
# input2 = realData[:,3]
# input3 = realData[:,4]
# input4 = realData[:,6]
# # print(len(input1))

# cor1 = cross_correlation(input1,input2)
# time2 = cross_correlation(input1,input3)
# time3 = cross_correlation(input1,input4)

# # Find index of maximal value for each of these time_n
# peak_index1 = np.argmax(cor1)
# # time1 = peak_index1/10000
# # MaxTime2 = np.max(time2)
# # IndmaxD2 = np.where(time2 == MaxTime2)[0][0]
# # MaxTime3 = np.max(time3)
# # IndmaxD3 = np.where(time3 == MaxTime3)[0][0]

# print(time1)


import numpy as np

def cross_correlation(signal1, signal2):
    # Flip the second signal
    signal2_flipped = np.flip(signal2)
    # Calculate the cross-correlation
    cross_corr_result = np.correlate(signal1, signal2_flipped, mode='full')
    # Find the index of the peak correlation
    peak_index = np.argmax(cross_corr_result)
    return cross_corr_result, peak_index

def find_peak_indices(signal1, signal2, peak_index):
    # Length of the second signal
    signal2_len = len(signal2)
    # Length of the result, considering mode='full'
    result_len = len(signal1) + len(signal2) - 1
    adjusted_peak_index = peak_index - (len(signal2) - 1)
    return adjusted_peak_index

# Example signals
signal1 = np.array([1, 1, 1, 1, 1, 6])
signal2 = np.array([0, 0, 1, 0, 0, 0])


# Calculate cross-correlation and find peak index
result, peak_index = cross_correlation(signal1, signal2)
print("Cross-correlation result:", result)
print("Peak index:", peak_index)

# Find peak index in the original matrix
adjusted_peak_index = find_peak_indices(signal1, signal2, peak_index)
print("Adjusted peak index within the original matrix:", adjusted_peak_index)
print("Value in the original matrix where peak correlation occurs:", signal1[adjusted_peak_index])
    # Calculate the index range of the original matrix
#     original_index_start = peak_index
#     original_index_end = peak_index + signal2_len - 1
#     return original_index_start, original_index_end

# # Example signals
# signal1 = np.array([1, 2, 3, 4, 5])
# signal2 = np.array([0, 1, 2, 3, 4])

# # Calculate cross-correlation and find peak index
# result, peak_index = cross_correlation(signal1, signal2)
# print("Cross-correlation result:", result)
# print("Peak index:", peak_index)

# # Find peak indices in the original matrix
# original_index_start, original_index_end = find_peak_indices(signal1, signal2, peak_index)
# print("Peak occurs in the original matrix from index", original_index_start, "to", original_index_end)
# print("Values in the original matrix where peak correlation occurs:", signal1[original_index_start:original_index_end+1])
