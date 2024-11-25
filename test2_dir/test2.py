import pyxdf
import matplotlib.pyplot as plt
import numpy as np

print("input the dirname")
dirname = input()
filename = "./"+dirname+"/"+dirname+"data.xdf"

data, header = pyxdf.load_xdf(filename)
print("header",header)


number = 0

for stream in data:
    y = stream['time_series']
    stream_name = stream['info']['name'][0]

    print("stream name",stream_name,"\noriginal timestamps",stream['time_stamps'])
    clock_offset_ini = stream['footer']['info']['clock_offsets'][0]['offset'][0]['time'][0]
    print("clock offset",clock_offset_ini)
    print("namecheck",'Care' in stream_name)

    if('Care' in stream_name):
        #print("input start time")
        first_timestamp=stream['time_stamps'][0]
        print("calc check",float(first_timestamp),float(clock_offset_ini))
        start_time_offset = float(clock_offset_ini) - float(first_timestamp)
        stream['time_stamps'] = [x + start_time_offset for x in stream['time_stamps']]
        print("new timestamp",stream['time_stamps'][:9])

    if isinstance(y, list):
        # list of strings, draw one vertical line for each marker
        for timestamp, marker in zip(stream['time_stamps'], y):
            plt.axvline(x=timestamp)
            print(f'Marker "{marker[0]}" @ {timestamp:.2f}s')
    elif isinstance(y, np.ndarray):
        # numeric data, draw as lines
        print("y",y)
        plt.plot(stream['time_stamps'], y)


        plt.savefig("./"+dirname+"/save_"+str(number)+".png")
        number=number+1
    else:
        raise RuntimeError('Unknown stream format')
