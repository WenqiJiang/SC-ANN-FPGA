"""
Given M HBM channels in total, and N channels that is actually needed,
    return the channel id that can balance the idle channels.

For example, M = 8, N = 4,
    channel 0, 2, 4, 6 should be in use, rather than 0, 1, 2, 3 which can lead to routing issues
"""
import numpy as np

class ChannelIterator:

    current_logical_channel = int(0)
    physical_channel_per_logical_channel = 0.0

    def __init__(self, total_channel_num=32, channel_in_use=1):
        
        self.physical_channel_per_logical_channel = total_channel_num / channel_in_use


    def get_next_channel_id(self):
        # an iterator

        current_physical_channel = int(np.ceil(self.current_logical_channel * self.physical_channel_per_logical_channel))
        self.current_logical_channel += 1

        return current_physical_channel
