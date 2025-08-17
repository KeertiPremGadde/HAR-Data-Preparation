from imu_data_reader import get_imu_data, get_capaimu_data
from matplotlib import pyplot as plt
import numpy as np

# experiment = "C:/Dev/dataset/bruno-2-10-10-2019/imu/shimmer_imu/"
experiment = "C:/Dev/dataset/bruno-2-10-10-2019/shimmer_post/"
# source_mac_addr = "MT_imu_top_left_feet.log"
source_mac_addr = "SH_imu_left_feet.log"
source_mac_addr = "SH_imu_left_upper_arm.log"
source_mac_addr = "SH_imu_right_upper_arm.log"


experiment = "./"
#source_mac_addr = "Star_P1_Right.csv"
source_mac_addr = "P1_Right.csv"
# source_mac_addr = "MT_imu_top_left_feet.log"

# source_mac_addr = "P1_imu_left_feet.log"
# source_mac_addr = "P1_imu_upper_right_arm.log"i
# source_mac_addr = "P1_imu_upper_left_arm.log"

data = get_capaimu_data(experiment, source_mac_addr)
# data = get_imu_data(experiment, source_mac_addr)


fig, ax = plt.subplots( figsize=(16, 4))
plt.title(source_mac_addr)


ax.plot(data[:, 0], data[:, 7], color='r')
ax.plot(data[:, 0], data[:, 8], color='g')
ax.plot(data[:, 0], data[:, 9], color='b')

ax.scatter(data[:, 0], data[:, 7], color='r', alpha=0.5)
ax.scatter(data[:, 0], data[:, 8], color='g', alpha=0.5)
ax.scatter(data[:, 0], data[:, 9], color='b', alpha=0.5)


# ax.plot(data[:, 0], data[:, 17])
# ax.plot(data[:, 0], data[:, 18])
# ax.plot(data[:, 0], data[:, 19])




selection_start = None
selection_end = None
current_line_start = None
current_line_end = None
current_selection = None


def onclick(event):
    global selection_start, selection_end, ax, current_selection, current_line_start, current_line_end
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

    if selection_start is None:
        selection_start = event.xdata
    elif selection_end is None:
        selection_end = event.xdata
    else:
        selection_end = None
        selection_start = event.xdata

    if selection_end is not None:
        if current_selection is None:
            current_selection = ax.axvspan(selection_start, selection_end, alpha=0.3)
        else:
            polygon = [[selection_start, 0], [selection_start, 1], [selection_end, 1], [selection_end, 0], [selection_start, 0]]
            current_selection.set_xy(polygon)

        if current_line_end is None:
            current_line_end = ax.axvline(selection_end)
        else:
            current_line_end.set_xdata(selection_end)
    else:
        if current_line_start is None:
            current_line_start = ax.axvline(selection_start)
        else:
            current_line_start.set_xdata(selection_start)

    plt.draw()
    print(selection_start, selection_end)
    try:
        print("len", selection_end - selection_start,
              (selection_end - selection_start) / 60.0 )
    except:
        pass


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()

print(selection_start, selection_end, selection_end - selection_start)