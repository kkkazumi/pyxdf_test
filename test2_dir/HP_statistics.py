import pyxdf
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import scipy.stats
import math
import csv

VITAL_SYS_CHANNEL=11
VITAL_HR_CHANNEL=14
SV=70

#graph drawing
GRAPH_VER = 1
GRAPH_HOR = 3
GRAPH_SYS = 1
GRAPH_CO = 2
GRAPH_SVR = 3

ROT_THETA=-45

def extract(X,t):
    array = np.zeros_like(X.T)
    array[:,0]=t
    array[:,1]=X[0]

    extracted = array[(array[:, 0] >= 0) & (array[:, 0] <= 60)]
    return extracted

def calculate_centroid(X, Y):
    centroid_X = np.mean(X)
    centroid_Y = np.mean(Y)
    return centroid_X, centroid_Y

def rotate_45_degrees(X, Y):
    theta = np.radians(ROT_THETA)
    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    points = np.vstack((X, Y))
    rotated_points = np.dot(rotation_matrix, points)
    
    return rotated_points

def get_ratio(NH_data,RB_data,HM_data,baseline):
    return np.log(np.array(NH_data)/baseline),np.log(np.array(RB_data)/baseline),np.log(np.array(HM_data)/baseline)

def check_marker(org_marker,filename,old_filename):
    new_marker = np.zeros((1,2))
    count=0
    if not os.path.exists(filename):
        new_marker = np.loadtxt(old_filename,delimiter=",")
        for i in new_marker:
            #print("check marker:",i)
            #input()
            if(i[1]==1010):
                i[0]=i[0]+76
            elif(i[1]==1050):
                i[0]=i[0]+136
            elif(i[1]==2010):
                i[0]=i[0]+76
            elif(i[1]==2050):
                i[0]=i[0]+136
            elif(i[1]==3010):
                i[0]=i[0]+76
            elif(i[1]==3050):
                i[0]=i[0]+136
            print("check marker:",i)
    else:
        new_marker = np.loadtxt(filename,delimiter=",")
    return new_marker


# Aの値が特定の範囲にある場合にBの値を抽出する関数
def extract_values(A, B):

    NH_BP = []
    RB_BP = []
    HM_BP = []

    tNH_BP = []
    tRB_BP = []
    tHM_BP = []

    for i in range(len(A) - 1):
        if A[i][1] == 1010 and A[i + 1][1] == 1500:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            NH_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
            tNH_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 0]-start_time)
        elif A[i][1] == 2010 and A[i + 1][1] == 2500:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            RB_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
            tRB_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 0]-start_time)
        elif A[i][1] == 3010 and A[i + 1][1] == 3500:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            HM_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
            tHM_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 0]-start_time)
    return NH_BP, RB_BP, HM_BP, tNH_BP, tRB_BP, tHM_BP

def attach_timestamp(org_array,timestamp_stream):
    new_array = np.zeros((org_array.shape[0],2))
    new_array[:,1] = org_array
    new_array[:,0]= timestamp_stream
    return new_array
                
#main

print("input the dirname")
dirname = input()
filename = "./"+dirname+"/"+dirname+"data.xdf"

data, header = pyxdf.load_xdf(filename)
print("header",header)


for stream in data:
    y = stream['time_series']
    stream_name = stream['info']['name'][0]
    #print("stream name",stream_name)
    clock_offset_ini = stream['footer']['info']['clock_offsets'][0]['offset'][0]['time'][0]

    #set timestamp
    if('Care' in stream_name):
        first_timestamp=stream['time_stamps'][0]
        start_time_offset = float(clock_offset_ini) - float(first_timestamp)
        stream['time_stamps'] = [x + start_time_offset for x in stream['time_stamps']]

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            #print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):

        if('vital' in stream_name): # numeric data, draw as lines
            if(not('vital2' in stream_name)):
                _co=SV*y[:,VITAL_HR_CHANNEL]
                _sys=y[:,VITAL_SYS_CHANNEL]
                _svr=_sys/_co
                svr=attach_timestamp(_svr,stream['time_stamps'])
                co=attach_timestamp(_co,stream['time_stamps'])
                sys=attach_timestamp(_sys,stream['time_stamps'])
        else:
            #plt.plot(stream['time_stamps'], y)
            marker=attach_timestamp(y[:,0],stream['time_stamps'])
            #print("marker data",marker)
            filename = './'+dirname+'/new_taskmarker.csv'
            old_filename= './'+dirname+'/new_marker.csv'
            marker=check_marker(marker,filename,old_filename)
            np.savetxt(filename, marker, delimiter=',')


    else:
        raise RuntimeError('Unknown stream format')

print("co",sum(co[:50,1])/50)
print("svr(tpr)",sum(svr[:50,1])/50)
    
co_baseline = sum(co[:50,1])/50
svr_baseline = sum(svr[:50,1])/50

NH_CO, RB_CO, HM_CO ,tNH_CO, tRB_CO, tHM_CO = extract_values(marker, co)
NH_SVR, RB_SVR, HM_SVR, tNH_SVR, tRB_SVR, tHM_SVR = extract_values(marker, svr)

NH_CO_r,RB_CO_r,HM_CO_r = get_ratio(NH_CO,RB_CO,HM_CO,co_baseline)
NH_SVR_r,RB_SVR_r,HM_SVR_r = get_ratio(NH_SVR,RB_SVR,HM_SVR,svr_baseline)

rotated_NH = rotate_45_degrees(NH_SVR_r,NH_CO_r)
rotated_RB = rotate_45_degrees(RB_SVR_r,RB_CO_r)
rotated_HM = rotate_45_degrees(HM_SVR_r,HM_CO_r)
#plt.plot(tNH_SVR,rotated_NH[0],label="nohint",linestyle=":")
#plt.plot(tRB_SVR,rotated_RB[0],label="robot",linestyle="-.")
#plt.plot(tHM_SVR,rotated_HM[0],label="human")

extracted_NH = extract(rotated_NH,tNH_SVR)
extracted_RB = extract(rotated_RB,tRB_SVR)
extracted_HM = extract(rotated_HM,tHM_SVR)

print("RB mean",np.mean(extracted_RB[:,1])\
,",RB std",np.std(extracted_RB[:,1])\
,",RB max",np.max(extracted_RB[:,1])\
,",RB min",np.min(extracted_RB[:,1]))

print("HM mean",np.mean(extracted_HM[:,1])\
,",HM std",np.std(extracted_HM[:,1])\
,",HM max",np.max(extracted_HM[:,1])\
,",HM min",np.min(extracted_HM[:,1]))

with open('./'+dirname+"/"+ dirname+'_y_valuesRB.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['y_values'])
    for value in extracted_RB[:,1]:
        writer.writerow([value])

with open('./'+dirname+"/"+ dirname+'_y_valuesHM.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['y_values'])
    for value in extracted_HM[:,1]:
        writer.writerow([value])

#plt.xlim(0,60)
#plt.ylim(-1,1)
#plt.xlabel("time [s]")
#plt.ylabel("HP")
#plt.legend()

# 値yの配列を抽出します
#y_values1 = extracted_RB[:, 1]
#y_values2 = extracted_HM[:, 1]

# 箱ひげ図を作成します
#plt.boxplot([y_values1, y_values2], vert=True, patch_artist=True, labels=['Robot', 'Human'])

# y軸のラベルを変更します
#plt.ylabel('Hemodynamic Profile')
#plt.grid(True)

# 凡例を追加します
#plt.legend(['Robot', 'Human'])

#plt.savefig("./"+dirname+"/"+dirname+"_HP_statistics.png")
