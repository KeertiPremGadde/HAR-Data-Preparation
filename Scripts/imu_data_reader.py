import os
import json
import re
import numpy as np
import pandas as pd
# from experiment_handler.time_synchronisation import convert_timestamps
# from experiment_handler.finder import find_all_imu_files


def load_imu_file(filepath):
    lines = []
    with open(filepath, 'r') as file:
        try:
            lines = file.read().split("\n")
        except UnicodeDecodeError as e:
            print(e)

    return lines


def _get_beacon_id(ble_data, use_uuid=True):
    if use_uuid:
        return ble_data['uuid'] + "-" + str(ble_data['major']) + "-" + str(ble_data['minor'])
    else:
        return ble_data['macAdd']


def find_and_categorize_beacon_ids(experiments, threshold=45, save_ids=True):
    """
    Find bluetooth beacon ids and count how many times they occur during the given experiments
    Observation showed that some beacon ids randomly appear only a few times during the experiments
    This function sort the ids into two categories (static id: appearing again and again, changing ids: occuring only a few times)


    Parameters
    ----------
    experiments: list of str
        List of pathes to the experiment roots
    threshold: int
        Defines separation threshold. Beacon ids with equal or less then this many detection are defined as changing id,
    save_ids: boolean
        If True result dictionary is saved to 'beacon_ids.json'

    Returns
    -------
        static_ids: list of str
            IDs with more detection than the threshold
        changing_ids: list of str
            IDs with less or equal detection than the threshold
        ids_detection_counts: dict
            Dictionary with id as key and count of detection as value

    """
    # Recompute set of ids from bluetooth beacons which are apparently static
    ids_detection_counts = {}

    for exp_root in experiments:
        imu_files = find_all_imu_files(exp_root)
        for filepath in imu_files:
            imu_lines = load_imu_file(filepath)

            for imu_line in imu_lines:
                try:
                    data = json.loads(imu_line)
                except json.decoder.JSONDecodeError:
                    continue

                if data['type'] != 'ble':
                    continue

                for beacon in data['beacons']:
                    current_id = _get_beacon_id(beacon)

                    if current_id in ids_detection_counts.keys():
                        ids_detection_counts[current_id] += 1
                    else:
                        ids_detection_counts[current_id] = 1

    static_ids = []
    changing_ids = []

    for key in ids_detection_counts.keys():
        if ids_detection_counts[key] > threshold:
            static_ids.append(key)
        else:
            changing_ids.append(key)

    # save ids into file
    if save_ids:
        sorted_ids = {
            "static_ids": static_ids,
            "changing_ids": changing_ids
        }
        with open('beacon_ids.json', 'w') as fp:
            json.dump(sorted_ids, fp)

    return static_ids, changing_ids, ids_detection_counts


def load_beacon_ids():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'beacon_ids.json'), 'r') as fp:
        sorted_ids = json.load(fp)
        return sorted_ids


def get_ble_data(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read beacon data for a given source (e.g. P3 left hand) in a time interval

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    source: str
        Name of the IMU file without extension e.g. P3_imu_right
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values

    Returns
    -------
        parsed_data_rssi: numpy array
            beacon's rssi data with columns order:  <timestamp>, <rssi value of unique id 0>, ...
        parsed_data_tx: numpy array
            beacon's tx data with columns order:  <timestamp>, <tx value of unique id 0>, ...
        unique_ids: list of strings
            Containing unique beacon ids in the same order as the columns above (offseted by timestamp)
    """

    filepath = os.path.join(experiment_path, "imu", source + ".log")
    imu_lines = load_imu_file(filepath)

    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

    ids = load_beacon_ids()
    unique_ids = ids['static_ids']
    parsed_data_tx = np.zeros((0, len(unique_ids) + 1))
    parsed_data_rssi = np.zeros((0, len(unique_ids) + 1))

    # Parse lines:
    for imu_line in imu_lines:
        try:
            data = json.loads(imu_line)
        except json.decoder.JSONDecodeError:
            continue

        if data['type'] != 'ble':
            continue

        new_data_entry_tx = np.zeros((1, len(unique_ids) + 1))
        new_data_entry_rssi = np.zeros((1, len(unique_ids) + 1))

        new_data_entry_rssi[0, 0] = data['time']
        new_data_entry_tx[0, 0] = data['time']

        for beacon in data['beacons']:
            current_id = _get_beacon_id(beacon)
            if current_id in unique_ids:
                column_index = unique_ids.index(current_id)
                new_data_entry_rssi[0, column_index + 1] = beacon['rssi']
                new_data_entry_tx[0, column_index + 1] = beacon['tx']
        parsed_data_tx = np.append(parsed_data_tx, new_data_entry_tx, axis=0)
        parsed_data_rssi = np.append(parsed_data_rssi, new_data_entry_rssi, axis=0)

    if start is not None:
        parsed_data_tx = parsed_data_tx[parsed_data_tx[:, 0] >= start_timestamp, :]
        parsed_data_rssi = parsed_data_rssi[parsed_data_rssi[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data_tx = parsed_data_tx[parsed_data_tx[:, 0] <= end_timestamp, :]
        parsed_data_rssi = parsed_data_rssi[parsed_data_rssi[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data_rssi[:, 0] = convert_timestamps(experiment_path, parsed_data_rssi[:, 0], imu_reference_time,
                                               reference_time)
        parsed_data_tx[:, 0] = convert_timestamps(experiment_path, parsed_data_tx[:, 0], imu_reference_time,
                                               reference_time)

    return parsed_data_rssi, parsed_data_tx, unique_ids


def get_imu_data_igroups(experiment_path, source, start=None, end=None, reference_time=None, convert_time=True):
    """
    Read imu data for a given source (e.g. P3 left hand) in a time interval
    
    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    source: str
        Name of the IMU file without extension e.g. P3_imu_right
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values
    use_pkl: bool
        If set the pickle serialized binary file will be read instead of the .log text file. (If the file doesn't exists yet, the .log file is loaded and then the .pkl file is created.)

    Returns
    -------
        parsed_data: numpy array
            IMU data with columns order:  <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>, <yaw>, <qx>, <qy>, <gz>, <qw>
    """
    npy_filepath = os.path.join(experiment_path, "imu", source + "_movement-data.npy")

    if not os.path.exists(npy_filepath):
        log_filepath = os.path.join(experiment_path, "imu", source + ".log")
        parsed_data = create_imu_log_file_movement_data(log_filepath, npy_filepath)
    else:
        parsed_data = np.load(npy_filepath)

    # Convert start and end time:
    imu_reference_time = source.split("_")[0] + "_IMU"
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

    if start is not None:
        parsed_data = parsed_data[parsed_data[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data = parsed_data[parsed_data[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], imu_reference_time, reference_time)

    return parsed_data


def convert_xsens_and_create_npy(filename, npy_filename):
    """
    Parameters
    ----------

    experiment_path: path to the MT_*.txt

    :param experiment_path:
    :return: A
        IMU data with columns order:
        <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>,
            <yaw>, <qx>, <qy>, <gz>, <qw> <lin_x> <lin_y> <lin_z> <gyr_limb> <lin_x_snsr> <lin_y_snsr> <lin_z_snsr>
    """
    """
        0 PackageCount, 14-16 Acc_(X/Y/Z), 17-19 LinAcc(X/Y/Z), 20-22 Gyro(X/Y/Z), 23-25 Mag(X/Y/Z),
        34 Roll, 35 Pitch, 36 Yaw
    """


    # Determine frequency.
    sampl_freq_hz = None
    with open(filename, "r") as f_imu_data:
        for line in  f_imu_data:
            if "Update Rate:" in line:
                sampl_freq_hz = float(re.search(r'\d+.\d', line).group(0))
                break





    data_raw = np.loadtxt(filename, comments="//", skiprows=5, delimiter=',',
                   usecols=[0,
                            14, 15, 16, # acc
                            17, 18, 19, # lin acc
                            20, 21, 22, # gyr
                            23, 24, 25, # mag
                            34, 35, 36, # euler
                            #29, 30, 31, 32 # quaternions (special OriInc)
                            ]
    )

    data_old_way = np.zeros( (len(data_raw), 21 + 3) )
    # Time is now in real time instead of package count. This is our best time estimate with this firmware.
    t_step = 1.0 /  float(sampl_freq_hz)

    data_old_way[:, 0] = np.arange(start=0.0, stop=len(data_raw)*t_step, step=t_step)

    # The accelerometer.
    data_old_way[:, [1,2,3] ] = data_raw[:, [1,2,3] ]

    # The gyroscope.
    data_old_way[:, [4, 5, 6]] = data_raw[:, [7, 8, 9]]

    # The magnetometer.
    data_old_way[:, [7, 8, 9]] = data_raw[:, [10, 11, 12]]

    # Roll Pitch Yaw
    data_old_way[:, [10, 11, 12]] = data_raw[:, [13, 14, 15]]

    # quaternions
    # data_old_way[:, [13, 14, 15, 16]] = data_raw[:, [16, 17, 18, 19]]

    # Linear accelerometer.
    data_old_way[:, [17, 18, 19]] = data_raw[:, [4, 5, 6]]

    # The gyro on the limb axis. In the case of xsens, the X points to the body.
    data_old_way[:, 20] = data_old_way[:, 4]

    # The linear acceleration in the sensor frame of reference.
    def rotate_to_snsr_frame(acc, roll, pitch, yaw):
        rot_matrix = np.array([
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
                np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
            ],

            [
                np.cos(pitch) * np.sin(yaw),
                np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
                np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
            ],

            [
                -np.sin(pitch),
                np.sin(roll) * np.cos(pitch),
                np.cos(roll) * np.cos(pitch)
            ]
        ])
        rot_matrix = np.linalg.inv(rot_matrix)
        x = np.array([acc])
        v = rot_matrix.dot(x.T)
        return v.flat
    for i in range( len(data_old_way) ):
        acc = data_old_way[i, [17, 18, 19]].T
        data_old_way[i, [21, 22, 23]] = rotate_to_snsr_frame(acc, \
                                                    data_old_way[i][10], data_old_way[i][11], data_old_way[i][12])


    # Save.
    np.save(npy_filename, data_old_way)

    return data_old_way

def convert_shimmer_and_create_npy(filename, npy_filename, acc_type="WR"):


    # Figure out what is the separator from the first line
    with open(filename, 'r') as f:
        str_sep = f.readline().split("=")[1].split("\"")[0]

    data_raw =  pd.read_csv(filename, sep=str_sep, skiprows=[0,2], header=0)

    # Remove sensor specific name information.
    map_cols_to_rename = {}
    for col in data_raw.columns:
        if col.startswith("Shimmer_"):
            map_cols_to_rename[col] = "_".join(col.split("_")[2:])
    data_raw.rename(index=str, columns=map_cols_to_rename, inplace=True)

    data_old_way = np.zeros((len(data_raw), 21 + 3))
    # timestamps from ms to seconds.
    try:
        data_old_way[:, 0] = data_raw[u'TimestampSync_Unix_CAL'] / 1000.0
    except:
        data_old_way[:, 0] = data_raw[u'Timestamp_Unix_CAL'] / 1000.0

    # The accelerometer.
    data_old_way[:, 1] = data_raw[u'Accel_'+acc_type+u'_Y_CAL']
    data_old_way[:, 2] = -1.0*data_raw[u'Accel_'+acc_type+u'_X_CAL']
    data_old_way[:, 3] = data_raw[u'Accel_'+acc_type+u'_Z_CAL']

    # The gyroscope.
    deg_to_rads_const = np.pi / 180.0
    data_old_way[:, 4] = data_raw[u'Gyro_Y_CAL'] * deg_to_rads_const
    data_old_way[:, 5] = -1.0*data_raw[u'Gyro_X_CAL'] * deg_to_rads_const
    data_old_way[:, 6] = data_raw[u'Gyro_Z_CAL'] * deg_to_rads_const

    # The magnetometer.
    data_old_way[:, 7] = data_raw[u'Mag_X_CAL']
    data_old_way[:, 8] = data_raw[u'Mag_Y_CAL']
    data_old_way[:, 9] = data_raw[u'Mag_Z_CAL']

    # Quaternions
    # // 13-16
    qw = data_raw[u'Quat_Madge_9DOF_W_LN_CAL']
    data_old_way[:, 13] = qw
    qx = data_raw[u'Quat_Madge_9DOF_X_LN_CAL']
    data_old_way[:, 14] = qx
    qy = data_raw[u'Quat_Madge_9DOF_Y_LN_CAL']
    data_old_way[:, 15] = qy
    qz = data_raw[u'Quat_Madge_9DOF_Z_LN_CAL']
    data_old_way[:, 16] = qz

    # Roll Pitch Yaw
    # // 10, 11, 12
    roll = np.arctan2(2.0 * (qw * qx + qy * qz), (1.0 - 2 * (qx ** 2 + qy ** 2)))
    data_old_way[:, 10] = roll
    pitch = 2.0 * (qw * qy - qz * qx)
    for i in np.arange( len(pitch) ):
        if np.abs(pitch[i]) >= 1.0:
            pitch[i] = np.copysign(np.pi / 2.0, pitch[i])
        else:
            pitch[i] = np.arcsin(pitch[i])
    data_old_way[:, 11] = pitch
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), (1.0 - 2 * (qy ** 2 + qz ** 2)))
    data_old_way[:, 12] = yaw

    # Generate Linear acceleration using orientation.
    # // 17, 18, 19 | 21, 22, 23 for in sensor frame.
    def remove_gravity_euler(acc, roll, pitch, yaw):
        rot_matrix = np.array([
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
                np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
            ],

            [
                np.cos(pitch) * np.sin(yaw),
                np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
                np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
            ],

            [
                -np.sin(pitch),
                np.sin(roll) * np.cos(pitch),
                np.cos(roll) * np.cos(pitch)
            ]
        ])
        x = np.array([acc])
        v = rot_matrix.dot(x.T)
        g = np.array([[0.0, 0.0, 9.80665]])
        lin_sim_global = v - g.T

        rotated_back_to_snr = np.linalg.inv(rot_matrix).dot(lin_sim_global)
        x, y, z = rotated_back_to_snr.flat
        return lin_sim_global.flat, [y, -x, z]
    for i in range( len(data_old_way) ):
        acc = data_old_way[i, [1,2,3]].T
        lin_sim_global, rotated_back_to_snr = remove_gravity_euler(acc, roll[i], pitch[i], yaw[i])
        data_old_way[i, [17,18,19]] = rotated_back_to_snr
        data_old_way[i, [21, 22, 23]] = lin_sim_global

    # The gyro on the limb axis. In the case of shimmer, the Y points to the body.
    data_old_way[:, 20] = data_old_way[:, 5]

    # Save.
    np.save(npy_filename, data_old_way)

    return data_old_way




def get_capaimu_data(experiment_path, mac_addr, start=None, end=None, reference_time=None, convert_time=False):
    
    filenames = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if mac_addr in f and f.split('.')[-1] == 'csv']

    print(filenames)
    if len(filenames) < 1:
        print("IMU file not found for ", mac_addr)
        exit()
    elif len(filenames) > 1:
        print("too many IMU files found for ", mac_addr)
        print(filenames)
        exit()

    filename = filenames[0]

    data_raw =  pd.read_csv(filename, sep=',')
    
    print(data_raw)

    data_old_way = np.zeros((len(data_raw), 11))
    # timestamps from ms to seconds.
    data_old_way[:, 0] = data_raw[u'Local_Time'] / 1000.0
    
    # The accelerometer.
    data_old_way[:, 1] = data_raw['A_x']
    data_old_way[:, 2] = data_raw['A_y']
    data_old_way[:, 3] = data_raw['A_z']

    # The gyroscope.
    data_old_way[:, 4] = data_raw['G_x']
    data_old_way[:, 5] = data_raw['G_y']
    data_old_way[:, 6] = data_raw['G_z']

    # The magnetometer.
    data_old_way[:, 7] = data_raw['M_x']
    data_old_way[:, 8] = data_raw['M_y']
    data_old_way[:, 9] = data_raw['M_z']

    # The capacitive
    data_old_way[:, 10] = data_raw['Cap']
    
    return data_old_way


def get_imu_data(experiment_path, mac_addr, start=None, end=None, reference_time=None, convert_time=False):
    """
    Read imu data for a given source in a time interval

    Parameters
    ----------
    experiment_path: str
        Root of the experiment (e.g. /data/igroups/experiment_8)
    mac_addr: str
        IMU's mac address
    start: float
        Return values from this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    end: float
        Return values until this timestamp (if reference_time is set, the value is interpreted as time on that channel)
    reference_time: str
        Use this signal channel's time for reference (convert start and end values to correspond with IMU time)
    convert_time: bool
        If set the returned array will contain timestamp in reference_time's values
    use_pkl: bool
        If set the pickle serialized binary file will be read instead of the .log text file. (If the file doesn't exists yet, the .log file is loaded and then the .pkl file is created.)

    Returns
    -------
        parsed_data: numpy array
            IMU data with columns order:  <timestamp>, <ax>, <ay>, <az>, <gx>, <gy>, <gz>, <mx>, <my>, <mz>, <roll>, <pitch>,
            <yaw>, <qx>, <qy>, <gz>, <qw> <lin_x> <lin_y> <lin_z> <gyr_limb> <lin_x_snsr> <lin_y_snsr> <lin_z_snsr>
    """
    imu_dir = os.path.join(experiment_path, "imu")
    filenames = [os.path.join(imu_dir, f) for f in os.listdir(imu_dir) if mac_addr in f and f.split('.')[-1] == 'log']

    if len(filenames) < 1:
        print("IMU file not found for ", mac_addr)
        exit()
    elif len(filenames) > 1:
        print("too many IMU files found for ", mac_addr)
        print(filenames)
        exit()
    filename = filenames[0]
    # print(filename)
    npy_filename = filename.split('.')[0] + "_movement-data.npy"

    if not os.path.exists(npy_filename):
        str_pref = os.path.split(filename)[-1].split("_")[0]
        if str_pref == "MT":
            parsed_data = convert_xsens_and_create_npy(filename, npy_filename)
        elif str_pref == "SH":
            parsed_data = convert_shimmer_and_create_npy(filename, npy_filename)
        else:
            parsed_data = create_imu_log_file_movement_data(filename, npy_filename, fix_jump=True)
    else:
        parsed_data = np.load(npy_filename)

    # Convert start and end time:
    imu_reference_time = mac_addr
    if start is not None:
        start_timestamp = convert_timestamps(experiment_path, start, reference_time, imu_reference_time)
    if end is not None:
        end_timestamp = convert_timestamps(experiment_path, end, reference_time, imu_reference_time)

    if start is not None:
        parsed_data = parsed_data[parsed_data[:, 0] >= start_timestamp, :]
    if end is not None:
        parsed_data = parsed_data[parsed_data[:, 0] <= end_timestamp, :]

    if convert_time:
        parsed_data[:, 0] = convert_timestamps(experiment_path, parsed_data[:, 0], imu_reference_time, reference_time)

    return parsed_data


def create_imu_log_file_movement_data(log_file_path, npy_file_path, fix_jump=False):
    imu_lines = load_imu_file(log_file_path)

    parsed_data = np.zeros((len(imu_lines), 20))
    parsed_data.fill(np.nan)

    # Parse lines:
    for i in range(len(imu_lines)):
        imu_line = imu_lines[i]
        try:
            data = json.loads(imu_line)
        except ValueError:
            continue

        if data['type'] != 'imu-bosch':
            continue

        parsed_data[i, 0] = data['time']
        parsed_data[i, 1] = data['measurement']['ax']
        parsed_data[i, 2] = data['measurement']['ay']
        parsed_data[i, 3] = data['measurement']['az']

        parsed_data[i, 4] = data['measurement']['gx']
        parsed_data[i, 5] = data['measurement']['gy']
        parsed_data[i, 6] = data['measurement']['gz']

        parsed_data[i, 7] = data['measurement']['mx']
        parsed_data[i, 8] = data['measurement']['my']
        parsed_data[i, 9] = data['measurement']['mz']

        parsed_data[i, 10] = data['measurement']['roll']
        parsed_data[i, 11] = data['measurement']['pitch']
        parsed_data[i, 12] = data['measurement']['yaw']

        parsed_data[i, 13] = data['measurement']['qx']
        parsed_data[i, 14] = data['measurement']['qy']
        parsed_data[i, 15] = data['measurement']['qz']
        parsed_data[i, 16] = data['measurement']['qw']

        parsed_data[i, 17] = data['measurement']['lin_x']
        parsed_data[i, 18] = data['measurement']['lin_y']
        parsed_data[i, 19] = data['measurement']['lin_z']

    parsed_data = parsed_data[~np.isnan(parsed_data).any(axis=1)]

    # If there is a real jump, fix it.
    if fix_jump:
        all_diffs = np.diff(parsed_data[:, 0])
        index_jump = np.argmax(all_diffs)
        jump_val = all_diffs[index_jump]
        # A jump is real if it is bigger than the 10 times the mean before and after it summed.
        mean_before, mean_after = np.mean(all_diffs[0:index_jump]), np.mean(all_diffs[index_jump+1:])
        if abs(jump_val) > (mean_before + mean_after)*10.0:
            # The jump value contains some positive real time passing, estimated by the mean after.
            real_jump = jump_val - mean_after
            print("Fixing jump of ", real_jump, "mean after it is", mean_after)
            parsed_data[0:index_jump+1, 0] += (real_jump if real_jump > 0 else -real_jump)

    np.save(npy_file_path, parsed_data)
    return parsed_data


def debug_orient():

    import pandas as pd

    # Assumes they are in radians.
    # def get_rot_mtx_to_rgs_cs(roll, pitch, yaw):
    #     rot_matrix = np.array([
    #         [
    #             np.cos(pitch) * np.cos(yaw),
    #             np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
    #             np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
    #         ],
    #
    #         [
    #             np.cos(pitch) * np.sin(yaw),
    #             np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
    #             np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
    #         ],
    #
    #         [
    #             -np.sin(pitch),
    #             np.sin(roll) * np.cos(pitch),
    #             np.cos(roll) * np.cos(pitch)
    #         ]
    #     ])
    #
    #     inv_rot = np.linalg.inv(rot_matrix)
    #     g =  np.array([ [0.0, 0.0, 9.80665]])
    #     gravity_in_sensor_frame = inv_rot.dot(g.T)
    #     return gravity_in_sensor_frame.flat


    def remove_gravity_euler(ax, ay, az, roll, pitch, yaw):
        rot_matrix = np.array([
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
                np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
            ],

            [
                np.cos(pitch) * np.sin(yaw),
                np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
                np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
            ],

            [
                -np.sin(pitch),
                np.sin(roll) * np.cos(pitch),
                np.cos(roll) * np.cos(pitch)
            ]
        ])
        x = np.array([ [ax, ay, az]])
        v = rot_matrix.dot(x.T)
        g = np.array([[0.0, 0.0, 9.80665]])
        return (v - g.T).flat

    def remove_gravity_quat(ax, ay, az, qw, qx, qy, qz):
        # rot_matrix = np.array([
        #     [
        #         2.0 * q0 ** 2 + 2.0 * q1 ** 2.0 - 1.0,
        #         2.0 * q1 * q2 - 2.0 * q0 * q3,
        #         2.0 * q1 * q3 + 2.0 * q0 * q2
        #     ],
        #
        #     [
        #         2.0 * q1 * q2 + 2.0 * q0 * q3,
        #         2.0 * q0 ** 2.0 + 2.0 * q2 ** 2.0 - 1.0,
        #         2.0 * q2 * q3 - 2.0 * q0 * q1
        #     ],
        #
        #     [
        #         2.0 * q1 * q3 - 2.0 * q0 * q2,
        #         2.0 * q2 * q3 + 2.0 * q0 * q1,
        #         2.0 * q0 ** 2.0 + 2.0 * q3 ** 2.0 - 1.0
        #     ]
        # ])
        roll = np.arctan2( 2.0*(qw*qx + qy*qz), (1.0 - 2*(qx**2 + qy**2))  )
        pitch = 2.0*(qw*qy - qz*qx)
        if np.abs(pitch) >= 1.0:
            pitch = np.copysign(np.pi / 2.0, pitch)
        else:
            pitch = np.arcsin(pitch)

        yaw = np.arctan2( 2.0*(qw*qz + qx*qy), (1.0 - 2*(qy**2 + qz**2))   )
        rot_matrix = np.array([
            [
                np.cos(pitch) * np.cos(yaw),
                np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw),
                np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)
            ],

            [
                np.cos(pitch) * np.sin(yaw),
                np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw),
                np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)
            ],

            [
                -np.sin(pitch),
                np.sin(roll) * np.cos(pitch),
                np.cos(roll) * np.cos(pitch)
            ]
        ])

        # x = np.array([[ax, ay, az]])
        # g = np.array([[0.0, 0.0, 9.80665]])
        # # return (x.T - np.linalg.inv(rot_matrix).dot(g.T) ).flat
        # # return x.flat
        # return (rot_matrix.dot(x.T) - g.T).flat


        #
        x = np.array([[ax, ay, az]])
        g = np.array([[0.0, 0.0, 9.80665]])
        # return (x.T - np.linalg.inv(rot_matrix).dot(g.T) ).flat
        # return x.flat
        return (x.T - np.linalg.inv(rot_matrix).dot(g.T)).flat


        # x = np.array([ [ax, ay, az]])
        # v = rot_matrix.dot(x.T)
        # g = np.array([[0.0, 0.0, 9.80665]])
        # return (v - g.T).flat


    # def get_rot_using_quat(q0, q1, q2, q3):
    #     rot_matrix = np.array([
    #         [
    #             2*q0**2 + 2*q1**2 - 1.0,
    #             2*q1*q2 - 2*q0*q3,
    #             2*q1*q3 + 2*q0*q2
    #         ],
    #
    #         [
    #             2*q1*q2 + 2*q0*q3,
    #             2*q0**2 + 2*q2**2 - 1.0,
    #             2*q2*q3 - 2*q0*q1
    #         ],
    #
    #         [
    #             2*q1*q3 - 2*q0*q2,
    #             2*q2*q3 + 2*q0*q1,
    #             2*q0**2 + 2*q3**2 - 1.0
    #         ]
    #     ])
    #
    #     inv_rot = np.linalg.inv(rot_matrix)
    #     g = np.array([[0.0, 0.0, 9.80665]])
    #     gravity_in_sensor_frame = inv_rot.dot(g.T)
    #     return gravity_in_sensor_frame.flat


    s = '/home/vitor/git/skeleton-framework/data/drill_seed_movements/dnc_bru1_Bru_1/Bru_imu_right_wrist_raw.csv'
    dt = pd.read_csv(s)

    roll = dt["roll"] * (np.pi / 180.0)
    pitch = dt["pitch"] * (np.pi / 180.0)
    yaw = dt["yaw"] * (np.pi / 180.0)

    q0, q1, q2, q3 = dt["qx"], dt["qy"], dt["gz"], dt["qw"]


    my_lin = np.zeros( (len(dt), 3) )
    my_grav = np.zeros( (len(dt), 3) )

    for i in range( len(dt) ):
        # v = get_rot_mtx_to_rgs_cs(roll[i], pitch[i], yaw[i])
        # # v = get_rot_using_quat(q0[i], q1[i], q2[i], q3[i])
        # gx, gy, gz = v
        # my_grav[i] = [gx, gy, gz]
        # my_lin[i] = [dt["ax"][i] - gx, dt["ay"][i] - gy, dt["az"][i] - gz]
        # my_lin[i] = remove_gravity_euler(dt["ax"][i], dt["ay"][i], dt["az"][i], roll[i], pitch[i], yaw[i])

        my_lin[i] = remove_gravity_quat(dt["ax"][i], dt["ay"][i], dt["az"][i], q0[i], q1[i], q2[i], q3[i])





    t = dt["timestamp"]
    from matplotlib import pyplot as plt

    plt.plot(t, my_lin[:,0], "r--", alpha=0.5, label="my_linx")
    plt.plot(t, dt["lin_x"], "r", alpha=0.5, label="linx")

    plt.plot(t, dt["lin_y"], "g", alpha=0.5, label="liny")
    plt.plot(t, my_lin[:, 1], "g--", alpha=0.5, label="my_liny")

    plt.plot(t, dt["lin_z"], "b", alpha=0.5, label="linz")
    plt.plot(t, my_lin[:, 2], "b--", alpha=0.5, label="my_linz")

    # plt.plot(t, dt["lin_x"] + my_grav[:,0] , "r--", alpha=0.5, label="my_ax")
    # plt.plot(t, dt["ax"], "r", alpha=0.5, label="ax")
    #
    # plt.plot(t, dt["ay"], "g", alpha=0.5, label="ay")
    # plt.plot(t, dt["lin_y"] + my_grav[:,1], "g--", alpha=0.5, label="my_ay")
    #
    # plt.plot(t, dt["az"], "b", alpha=0.5, label="az")
    # plt.plot(t, dt["lin_z"] + my_grav[:,2], "b--", alpha=0.5, label="my_az")

    # plt.plot(t, np.sqrt( np.square(my_grav[:, 0]) +  np.square(my_grav[:, 1]) + np.square(my_grav[:, 2]) ) , "g", label="grav norm")
    # plt.plot(t, np.sqrt(np.square(my_lin[:, 0]) + np.square(my_lin[:, 1]) + np.square(my_lin[:, 2])), "b--",
    #          label="my lin norm")
    #
    # plt.plot(t, np.sqrt(np.square(dt["lin_x"]) + np.square(dt["lin_y"]) + np.square(dt["lin_z"])), "b",
    #          label="true lin norm")
    #

    # norm_error = np.sqrt(np.square(dt["lin_x"]) + np.square(dt["lin_y"]) + np.square(dt["lin_z"])) - \
    #              np.sqrt(np.square(my_lin[:, 0]) + np.square(my_lin[:, 1]) + np.square(my_lin[:, 2]))


    norm_error = np.sqrt( np.square(dt["lin_x"] - my_lin[:, 0])  + np.square(dt["lin_y"] - my_lin[:, 1])
                          + np.square(dt["lin_z"] - my_lin[:, 2] ))

    plt.plot(t, norm_error, "k",
             label="difference")

    plt.legend()
    plt.show()

    plt.hist(norm_error)
    plt.title(str(np.sum(norm_error)))
    plt.show()





if __name__ == '__main__':
    debug_orient()
    exit(0)


    if False:
        experiments = [
            "/Users/hevesi/ownCloud/Datasets/igroups_experiment_8",
            "/Users/hevesi/ownCloud/Datasets/igroups_experiment_9"
        ]
        good, bad, id_det_counts = find_and_categorize_beacon_ids(experiments)
        print(good)
        print("Bad:")
        print(bad)

    exp_root = "/Volumes/DataDrive/igroups_recordings/igroups_experiment_8"
    exp_root = "/Users/hevesi/ownCloud/Datasets/igroups_experiment_8"
    exp_root = "/Users/hevesi/Desktop/EX_1"

    #data = get_ble_data(exp_root, "P2_imu_head", start=1800, end=2100, reference_time="video", convert_time=True)
    #data = get_imu_data(exp_root, "P3_imu_left", start=1800, end=2368, reference_time="video", convert_time=True)
    # data = get_imu_data(exp_root, "fcdbb37ffcd4")
    data = get_imu_data(exp_root, "fcdbb37ffcd4", start=0, end=None, reference_time="video_1", convert_time=True)
    print(data)

    import matplotlib.pyplot as plt

    plt.plot(data[:, 0], data[:, 1])
    plt.plot(data[:, 0], data[:, 2])
    plt.plot(data[:, 0], data[:, 3])
    plt.show()

