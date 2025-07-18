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

ROT_THETA=45

def append(filepath,data):
    try:
        with open(filepath,"a",newline="") as file_object:
            writer = csv.writer(file_object)
            writer.writerow(data)
        return filepath
    except Exception as e:
        raise Exception(e)

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
                i[0]=i[0]
            elif(i[1]==1050):
                i[0]=i[0]+76
            elif(i[1]==2010):
                i[0]=i[0]
            elif(i[1]==2050):
                i[0]=i[0]+76
            elif(i[1]==3010):
                i[0]=i[0]
            elif(i[1]==3050):
                i[0]=i[0]+76
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
            filename = './'+dirname+'/new_taskmarker_pre.csv'
            old_filename= './'+dirname+'/new_marker.csv'
            marker=check_marker(marker,filename,old_filename)
            np.savetxt(filename, marker, delimiter=',')


    else:
        raise RuntimeError('Unknown stream format')

print("co",sum(co[:50,1])/50)
print("svr(tpr)",sum(svr[:50,1])/50)
    
co_baseline = sum(co[:50,1])/50
svr_baseline = sum(svr[:50,1])/50
print("co_baseline",co_baseline,"svr_baseline",svr_baseline)

NH_CO, RB_CO, HM_CO ,tNH_CO, tRB_CO, tHM_CO = extract_values(marker, co)
NH_SVR, RB_SVR, HM_SVR, tNH_SVR, tRB_SVR, tHM_SVR = extract_values(marker, svr)

NH_CO_r,RB_CO_r,HM_CO_r = get_ratio(NH_CO,RB_CO,HM_CO,co_baseline)
NH_SVR_r,RB_SVR_r,HM_SVR_r = get_ratio(NH_SVR,RB_SVR,HM_SVR,svr_baseline)

rotated_NH = rotate_45_degrees(NH_SVR_r,NH_CO_r)[0]
rotated_RB = rotate_45_degrees(RB_SVR_r,RB_CO_r)[0]
rotated_HM = rotate_45_degrees(HM_SVR_r,HM_CO_r)[0]

START_TIME=0
BEGIN_END_TIME=20
MID_END_TIME=40
END_END_TIME=60

NH_beginningHP = rotated_NH[(np.array(tNH_SVR)>=START_TIME)&(np.array(tNH_SVR)<=BEGIN_END_TIME)]
RB_beginningHP = rotated_RB[(np.array(tRB_SVR)>=START_TIME)&(np.array(tRB_SVR)<=BEGIN_END_TIME)]
HM_beginningHP = rotated_HM[(np.array(tHM_SVR)>=START_TIME)&(np.array(tHM_SVR)<=BEGIN_END_TIME)]

NH_midHP = rotated_NH[(np.array(tNH_SVR)>BEGIN_END_TIME)&(np.array(tNH_SVR)<=MID_END_TIME)]
RB_midHP = rotated_RB[(np.array(tRB_SVR)>BEGIN_END_TIME)&(np.array(tRB_SVR)<=MID_END_TIME)]
HM_midHP = rotated_HM[(np.array(tHM_SVR)>BEGIN_END_TIME)&(np.array(tHM_SVR)<=MID_END_TIME)]

NH_endHP = rotated_NH[(np.array(tNH_SVR)>MID_END_TIME)&(np.array(tNH_SVR)<=END_END_TIME)]
RB_endHP = rotated_RB[(np.array(tRB_SVR)>MID_END_TIME)&(np.array(tRB_SVR)<=END_END_TIME)]
HM_endHP = rotated_HM[(np.array(tHM_SVR)>MID_END_TIME)&(np.array(tHM_SVR)<=END_END_TIME)]

append("HP3part_data.csv",str(dirname))
#append("HP3part_data.csv",str(np.mean(NH_beginningHP))+","+str(np.mean(RB_beginningHP))+","+str(np.mean(HM_beginningHP)))
data_beg=[np.mean(NH_beginningHP),np.mean(RB_beginningHP),np.mean(HM_beginningHP)]
data_mid=[np.mean(NH_midHP),np.mean(RB_midHP),np.mean(HM_midHP)]
data_end=[np.mean(NH_endHP),np.mean(RB_endHP),np.mean(HM_endHP)]
append("HP3part_data.csv",data_beg)
append("HP3part_data.csv",data_mid)
append("HP3part_data.csv",data_end)
