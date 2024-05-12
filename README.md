Python tools for carla leaderboard 2.0 training dataset capturing and processing.

Beta version 0.0.1


## Setup

Data processing
```Shell
cd /home/zc/carla/lb2/leaderboard/data_tools
python capture_sensor_data_LMDrive.py
python get_list_file.py /home/zc/datasets/lb2_lmdrive
python batch_merge_data.py /home/zc/datasets/lb2_lmdrive
python batch_merge_measurements.py /home/zc/datasets/lb2_lmdrive
```