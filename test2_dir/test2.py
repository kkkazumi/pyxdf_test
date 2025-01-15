import pyxdf
import matplotlib.pyplot as plt
import numpy as np
import os

VITAL_SYS_CHANNEL=11
VITAL_HR_CHANNEL=14
SV=70

#graph drawing
GRAPH_VER = 1
GRAPH_HOR = 3
GRAPH_SYS = 1
GRAPH_CO = 2
GRAPH_SVR = 3


def check_marker(org_marker,filename):
    new_marker = np.zeros((1,2))
    count=0
    if not os.path.exists(filename):
        for i in org_marker:
            print("check marker:",i)
            print("select: 0)keep, 1)delete, 2)change")
            mode=int(input())
            if(mode==0):
                if(count==0):
                    new_marker[0]=i
                    count=1
                else:
                    new_marker=np.vstack((new_marker,i))
            elif(mode==2):
                print("input new value")
                new_val = int(input())
                i[1]=new_val
                if(count==0):
                    new_marker[0]=i
                    count=1
                else:
                    new_marker=np.vstack((new_marker,i))
        print("new marker",new_marker)
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
    print("stream name",stream_name)
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
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
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
            print("marker data",marker)
            filename = './'+dirname+'/new_marker.csv'
            marker=check_marker(marker,filename)
            np.savetxt(filename, marker, delimiter=',')


    else:
        raise RuntimeError('Unknown stream format')

    
plt.figure(figsize=(18,6))
plt.subplot(GRAPH_VER,GRAPH_HOR,GRAPH_SYS)
NH_SYS, RB_SYS, HM_SYS ,tNH_SYS, tRB_SYS, tHM_SYS = extract_values(marker, sys)
print("nh",len(NH_SYS),"rb",len(RB_SYS),"hm",len(HM_SYS))
#plt.savefig("./"+dirname+"/save.png")
plt.plot(tNH_SYS,NH_SYS,label="nohint")
plt.plot(tRB_SYS,RB_SYS,label="robot")
plt.plot(tHM_SYS,HM_SYS,label="human")
plt.xlabel("timestamp[sec]")
plt.ylabel("value (sys)")
plt.axvline(x=76,color="black",linestyle="--")
plt.axvline(x=136,color="black",linestyle="--")
plt.legend()

plt.subplot(GRAPH_VER,GRAPH_HOR,GRAPH_CO)
NH_CO, RB_CO, HM_CO ,tNH_CO, tRB_CO, tHM_CO = extract_values(marker, co)
print(len(NH_CO),len(RB_CO),len(HM_CO))
#plt.savefig("./"+dirname+"/save.png")
plt.plot(tNH_CO,NH_CO,label="nohint")
plt.plot(tRB_CO,RB_CO,label="robot")
plt.plot(tHM_CO,HM_CO,label="human")
plt.xlabel("timestamp[sec]")
plt.ylabel("value (co)")
plt.axvline(x=76,color="black",linestyle="--")
plt.axvline(x=136,color="black",linestyle="--")
plt.legend()

plt.subplot(GRAPH_VER,GRAPH_HOR,GRAPH_SVR)
NH_SVR, RB_SVR, HM_SVR, tNH_SVR, tRB_SVR, tHM_SVR = extract_values(marker, svr)
print(len(NH_SVR),len(RB_SVR),len(HM_SVR))
#plt.savefig("./"+dirname+"/save.png")
plt.plot(tNH_SVR,NH_SVR,label="nohint")
plt.plot(tRB_SVR,RB_SVR,label="robot")
plt.plot(tHM_SVR,HM_SVR,label="human")
plt.xlabel("timestamp[sec]")
plt.ylabel("value (svr)")
plt.axvline(x=76,color="black",linestyle="--")
plt.axvline(x=136,color="black",linestyle="--")
plt.legend()

plt.savefig("./"+dirname+"/"+dirname+"_taskdata.png")
