"""
This script define the RecorderParser, which is used to parse logs generated from carla.Client.start_recorder
"""

class RecorderParser():

    def __init__(self, log_file, current_frame) -> None:
        self.log = log_file
        self.frame = current_frame

    def get_location(self):
        pass

    def get_velocity(self, actor):
        pass
