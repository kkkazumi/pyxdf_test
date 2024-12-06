import pyxdf
import matplotlib.pyplot as plt
import numpy as np

VITAL_SYS_CHANNEL=11
VITAL_HR_CHANNEL=14
SV=70

#graph drawing
GRAPH_VER = 3
GRAPH_HOR = 1

def check_marker(org_marker):
    new_marker = np.zeros((1,2))
    count=0
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
    return new_marker

# Aの値が特定の範囲にある場合にBの値を抽出する関数
def extract_values(A, B):
    SP_BP = []
    RB_BP = []
    HM_BP = []
    for i in range(len(A) - 1):
        if A[i][1] == 1010 and A[i + 1][1] == 1500:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            SP_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
        elif A[i][1] == 200 and A[i + 1][1] == 250:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            RB_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
        elif A[i][1] == 300 and A[i + 1][1] == 350:
            start_time = A[i][0]
            end_time = A[i + 1][0]
            HM_BP.extend(B[(B[:, 0] > start_time) & (B[:, 0] < end_time), 1])
    return SP_BP, RB_BP, HM_BP

def attach_timestamp(org_array,timestamp_stream):
    new_array = np.zeros((org_array.shape[0],2))
    new_array[:,1] = org_array
    new_array[:,0]= timestamp_stream
    return new_array
                
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

                plt.subplot(GRAPH_VER,GRAPH_HOR,1)
                plt.plot(stream['time_stamps'], svr[:,1],label="svr")
                plt.legend()

                plt.subplot(GRAPH_VER,GRAPH_HOR,2)
                plt.plot(stream['time_stamps'], co[:,1],label="co")
                plt.legend()

                plt.subplot(GRAPH_VER,GRAPH_HOR,3)
                plt.plot(stream['time_stamps'], sys[:,1],label="sys")
                plt.legend()
        else:
            #plt.plot(stream['time_stamps'], y)

            marker=attach_timestamp(y[:,0],stream['time_stamps'])
            print("marker data",marker)
            #marker=check_marker(marker)
            #print("new marker data",marker)

    else:
        raise RuntimeError('Unknown stream format')

SP_BP, RB_BP, HM_BP = extract_values(marker, svr)
print(len(SP_BP),len(RB_BP),len(HM_BP))
plt.savefig("./"+dirname+"/save.png")
