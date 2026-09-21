from loguru import logger
import math


class AirborneObject:

    def __str__(self):
        return "AirborneObject#%s(num_frames=%s, planned=%s)" % (self.id, self.num_frames, self.planned)

    def __init__(self, entity):
        self.location = []
        self.angular_rate = {}
        self.id = entity['id']
        self._planned = None
        # self.dt = 1/10 # 10fps
        # self.cam_angular_res = 0.02750197 # deg/px
        # self.max_frame_count_before_reset = 1

    def register_location(self, obj_location):
        self.location.append(obj_location)
        
        if self.planned is None:
            self._planned = obj_location.planned

        # angular rate calcs
        # if len(self.location) > 1:
        #     # only compute if 2 consecutive frames
        #     if (obj_location.frame.frame - self.location[-2].frame.frame) <= self.max_frame_count_before_reset:
        #         prev_obj_location = self.location[-2].bb.get_center()
        #         curr_obj_location = obj_location.bb.get_center()

        #         dx = curr_obj_location[0] - prev_obj_location[0]
        #         dy = curr_obj_location[1] - prev_obj_location[1]
                
        #         vx = dx/self.dt
        #         vy = dy/self.dt

        #         angular_rate_ground_truth = self.cam_angular_res*math.sqrt(vx*vx + vy*vy)
                
        #         # save in a dict where key is the frame id and value is the ang rate
        #         self.angular_rate[obj_location.frame.frame] = angular_rate_ground_truth

    @property
    def planned(self):
        return self._planned

    @property
    def num_frames(self):
        return len(self.location)
    
    # def get_angular_rate_for_frame(self, frame_num):
    #     if frame_num in self.angular_rate.keys():
    #         return self.angular_rate[frame_num]
    #     else:
    #         return None
    
    # @property
    # def all_angular_rates(self):
    #     return self.angular_rate
    
    # @property
    # def angular_rate_avg(self):
    #     return sum(self.angular_rate)/len(self.angular_rate)
    
    

