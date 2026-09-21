# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 07:34:48 2025

@author: thoma
"""

import numpy as np
import matplotlib.pyplot as plt
import csv


class EncounterGeometry():
    
    def __init__(self, ownship_time, ownship_north_m, ownship_east_m, ownship_up_m, ownship_speed_m_per_s,
                 intruder_time, intruder_north_m, intruder_east_m, intruder_up_m, intruder_speed_m_per_s, intruder_hdg_rad, intruder_vz,
                 h_dist, v_dist, dist,
                 takeoff_position, encounter_mission, encounter_start_alt, target_dt, enc_len):
        self.ownship_time = ownship_time
        self.ownship_north_m = ownship_north_m
        self.ownship_east_m = ownship_east_m
        self.ownship_up_m = ownship_up_m
        self.ownship_speed_m_per_s = ownship_speed_m_per_s
        
        self.intruder_time = intruder_time
        self.intruder_north_m = intruder_north_m
        self.intruder_east_m = intruder_east_m
        self.intruder_up_m = intruder_up_m
        self.intruder_speed_m_per_s = intruder_speed_m_per_s
        self.intruder_hdg_rad = intruder_hdg_rad
        self.intruder_vz = intruder_vz
        
        self.h_dist = h_dist
        self.v_dist = v_dist
        self.dist = dist
        
        self.takeoff_position = takeoff_position
        self.mission = encounter_mission
        self.encounter_start_alt = encounter_start_alt
        self.target_dt = target_dt
        self.enc_len = enc_len
    
    def lerp(self, param_list, i, t):
        y0 = param_list[i]
        nxt_i = i+1
        if nxt_i < len(param_list): 
            y1 = param_list[nxt_i]
            yt = y0 + (y1-y0) * t
            return yt
        else:
            return y0


class EncounterSimLogger():
    
    def __init__(self, daa, test_id, results_home, encounter_idx, encounter_geometry):
        self.daa = daa
        self.test_id = test_id
        self.results_home = results_home
        self.encounter_idx = encounter_idx
        self.geo = encounter_geometry
        
        self._closest_approach = float("inf")
        self._tcpa = -1.0 # time of closest approach
        self._os_pos = []
        self._os_hdg_rad = []
        self._os_spd = []
        self._os_vspd = []
        self._in_pos = []
        self._in_hdg_rad = []
        self._in_spd = []
        self._in_vspd = []
        self._hmd = []
        self._vmd = []
        self._md = []
        self._dist = [] # desired _md
        self._times = []
        self._avoids = []
        self._debug_i = 0
        self._closest_i = 0
        self._skip_count = 0
        self._rejoins_i = []
        self._lwcs = []
        self._lwc_is = [] #indexes during lwc
        self._lwc_crit_i = 0
        self._lwc_hmd = float("inf")
        self._lwc_vmd = float("inf")
        self._first_guided = False
        self._first_guidance = []
        self._guidance_count = 0
        self._guidance_counts = []
        self._in_pos_detected = []
        self._in_pos_smoothed = []
        self._pitch_outs = [] # pitch servo2
        self._roll_outs = [] # roll servo1
        self._yaw_outs = [] # yaw servo4
        self._throttle_outs = [] # throttle servo3
        self._pitch_rad = []
        self._roll_rad = []
        self._yaw_rad = []
        self._pitch_rate_rad = []
        self._roll_rate_rad = []
        self._yaw_rate_rad = []
        self._last_guidances = []
        self._daidalus_os = []
        self._daidalus_in = []
        self._daidalus_response = []
        self._daidalus_times = []
        self._detections = []
        self._tracks = []
        self._nmacs = []
        self._nmac_is = [] #indexes during nmac
        self._nmac_hmd = float("inf")
        self._nmac_vmd = float("inf")
    
    def fill_empty(self, dictionary, row_length):
        for d in dictionary.items():
            if not d[1]:
                dictionary[d[0]] = np.zeros(row_length).tolist()

    def do_output(self, daa=True):
        #debug plotting
        _dt = np.diff(self._times)
        _os_norths, _os_easts, _os_alts = zip(*self._os_pos)
        _in_norths, _in_easts, _in_alts = zip(*self._in_pos)

        if daa:
            if len(self._in_pos_detected) > 0:
                _in_norths_detect, _in_easts_detect, _in_alts_detect = zip(*self._in_pos_detected)
            else:
                _in_norths_detect = []
                _in_easts_detect = []
                _in_alts_detect = []

            if len(self._in_pos_smoothed) > 0:
                _in_norths_sm, _in_easts_sm, _in_alts_sm = zip(*self._in_pos_smoothed)
            else:
                _in_norths_sm = []
                _in_easts_sm = []
                _in_alts_sm = []


        daa_string = ''
        if self.daa:
            daa_string = 'DAA_ON'
            if self.test_id != '':
                daa_string += '-' + self.test_id
        else:
            daa_string = 'DAA_OFF'
            if self.test_id != '':
                daa_string += '-' + self.test_id
        
        # Ownship uv vector
        _os_hdg_array = np.array(self._os_hdg_rad)
        ownship_u_vec = np.sin(_os_hdg_array)  # X component (East)
        ownship_v_vec = np.cos(_os_hdg_array)  # Y component (North)
        
        # Closest approach
        _closest_norths = [_os_norths[self._closest_i], _in_norths[self._closest_i]]
        _closest_easts = [_os_easts[self._closest_i], _in_easts[self._closest_i]]
        
        # Rejoin positions
        _rejoin_norths = []
        _rejoin_easts = []
        for rejoin_i in self._rejoins_i:
            _rejoin_norths.append(_os_norths[rejoin_i])
            _rejoin_easts.append(_os_easts[rejoin_i])
        
        # Create 2x2 subplot grid
        fig, axs = plt.subplots(2, 5, figsize=(10, 8), sharex='col')
        
        # Top-left: Ownship North vs Time
        axs[0, 0].plot(self._times, _os_norths, color='b') # simulated
        axs[0, 0].plot(self.geo.ownship_time, self.geo.ownship_north_m, color='k', linestyle='dotted') # desired
        axs[0, 0].set_title('Ownship North vs Time')
        axs[0, 0].set_ylabel('Coordinate (m)')
        axs[0, 0].grid(True)
        # _err_norths = np.abs(ownship_north_m - _os_norths)[1:] * _dt
        
        # Top-right: Ownship East vs Time
        axs[0, 1].plot(self._times, _os_easts, color='b') # simulated
        axs[0, 1].plot(self.geo.ownship_time, self.geo.ownship_east_m, color='k', linestyle='dotted') # desired
        axs[0, 1].set_title('Ownship East vs Time')
        axs[0, 1].grid(True)
        # _err_easts = np.abs(ownship_east_m - _os_easts)[1:] * _dt
        
        # Top-more-right: Ownship Alt vs Time
        axs[0, 2].plot(self._times, _os_alts, color='b') # simulated
        axs[0, 2].plot(self.geo.ownship_time, self.geo.ownship_up_m, color='k', linestyle='dotted') # desired
        axs[0, 2].set_title('Ownship Alt vs Time')
        axs[0, 2].set_ylim(0.0, np.max(self.geo.ownship_up_m) * 1.2)
        axs[0, 2].grid(True)
        # _err_alts = np.abs(ownship_up_m - _os_alts)[1:] * _dt
        
        # Top-even-more-right: Ownship Ground Speed vs Time
        axs[0, 3].plot(self._times, self._os_spd, color='b') # simulated
        axs[0, 3].plot(self.geo.ownship_time, self.geo.ownship_speed_m_per_s, color='k', linestyle='dotted') # desired
        axs[0, 3].set_title('Ownship Grd Spd  vs Time')
        axs[0, 3].set_ylabel('Ground Speed (m/s)')
        axs[0, 3].set_ylim(0.0, np.max(self.geo.ownship_speed_m_per_s) * 1.2)
        axs[0, 3].grid(True)
        # _err_spds = np.abs(ownship_speed_m_per_s - _os_spd)[1:] * _dt
        
        # Top-VERY-right: Horizontal Miss Distance vs Time
        axs[0, 4].plot(self._times, self._hmd, color='g') # simulated
        axs[0, 4].plot(self.geo.ownship_time, self.geo.h_dist, color='k', linestyle='dotted') # desired
        axs[0, 4].set_title('HMD vs Time')
        axs[0, 4].set_ylabel('Distance (m)')
        axs[0, 4].set_ylim(-np.max(np.abs(self._hmd)) * 1.2, np.max(np.abs(self._hmd)) * 1.2)
        axs[0, 4].grid(True)
        # _err_hmds = np.abs(h_dist - _hmd)[1:] * _dt
        
        # Bottom-left: Intruder North vs Time
        axs[1, 0].plot(self._times, _in_norths, color='r') # simulated
        axs[1, 0].plot(self.geo.intruder_time, self.geo.intruder_north_m, color='k', linestyle='dotted') # desired
        axs[1, 0].set_title('Intruder North vs Time')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Coordinate (m)')
        axs[1, 0].grid(True)
        
        # Bottom-right: Intruder East vs Time
        axs[1, 1].plot(self._times, _in_easts, color='r') # simulated
        axs[1, 1].plot(self.geo.intruder_time, self.geo.intruder_east_m, color='k', linestyle='dotted') # desired
        axs[1, 1].set_title('Intruder East vs Time')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].grid(True)
        
        # Bottom-more-right: Intruder Alt vs Time
        axs[1, 2].plot(self._times, _in_alts, color='r') # simulated
        axs[1, 2].plot(self.geo.intruder_time, self.geo.intruder_up_m, color='k', linestyle='dotted') # desired
        axs[1, 2].set_title('Intruder Alt vs Time')
        axs[1, 2].set_xlabel('Time (s)')
        axs[1, 2].set_ylim(0.0, np.max(self.geo.intruder_up_m) * 1.2)
        axs[1, 2].grid(True)
        
        # Bottom-even-more-right: Intruder Ground Speed vs Time
        axs[1, 3].plot(self.geo.intruder_time, self.geo.intruder_speed_m_per_s, color='k', linestyle='dotted') # desired
        axs[1, 3].set_title('Intruder Grd Spd vs Time')
        axs[1, 3].set_xlabel('Time (s)')
        axs[1, 3].set_ylabel('Ground Speed (m/s)')
        axs[1, 3].set_ylim(0.0, np.max(self.geo.intruder_speed_m_per_s) * 1.2)
        axs[1, 3].grid(True)
        
        # Bottom-VERY-right: Vertical Miss Distance vs Time
        axs[1, 4].plot(self._times, self._vmd, color='g') # simulated
        axs[1, 4].plot(self.geo.ownship_time, self.geo.v_dist, color='k', linestyle='dotted') # desired
        axs[1, 4].set_title('VMD vs Time')
        axs[1, 4].set_xlabel('Time (s)')
        axs[1, 4].set_ylabel('Distance (m)')
        axs[1, 4].set_ylim(-np.max(np.abs(self._vmd)) * 1.2, np.max(np.abs(self._vmd)) * 1.2)
        axs[1, 4].grid(True)
        # _err_vmds = np.abs(v_dist - _vmd)[1:] * _dt
        
        
        # Tidy up layout
        plt.tight_layout()
        fig.savefig(self.results_home + ("{}[{}]_out.png".format(daa_string, self.encounter_idx)),    # filename (extension sets format)
                    dpi=300,                # resolution in dots-per-inch
                    bbox_inches="tight",    # trim whitespace
                    transparent=False       # make background transparent
        )
        
        # init
        # err_score = -1
        # miss_score = (0,0,0,0) # tcpa, AbsMD, HMD, VMD
        
        # 3D miss distance
        _err_md = np.abs(np.array(self._dist) - np.array(self._md))[1:] * _dt
        err_score = np.sum(_err_md)
        # plt.figure()
        # plt.plot(self._times, self._md, color='g') # simulated
        # plt.plot(self.geo.ownship_time, self.geo.dist, color='k', linestyle='dotted') # desired
        # plt.xlabel('Time (s)')
        # plt.ylabel('Miss Distance 3D (m)')
        # plt.savefig(self.results_home + ("{}[{}]_m3d_out.png".format(daa_string, self.encounter_idx)),    # filename (extension sets format)
        #             dpi=300,                # resolution in dots-per-inch
        #             bbox_inches="tight",    # trim whitespace
        #             transparent=False       # make background transparent
        # )
        
        # MAP
        encounter_centre_tup = (np.sum(_closest_easts) / 2.0, np.sum(_closest_norths) / 2.0)
        encounter_centre = np.array(encounter_centre_tup)

        # # os_start = np.array(self.geo.ownship_east_m[0], self.geo.ownship_north_m[0])
        # # os_end = np.array(self.geo.ownship_east_m[-1], self.geo.ownship_north_m[-1])
        # # os_path = os_end - os_start
        # # os_path_len = np.linalg.norm(os_path) * 2.0
        # map_size = 15000
        # half_size = map_size / 2.0

        # os_max_north = np.max(_os_norths)
        # os_max_east = np.max(_os_easts)
        # os_min_north = np.min(_os_norths)
        # os_min_east = np.min(_os_easts)

        # if len(_in_norths_detect) > 0:
        #     in_max_north = np.max(_in_norths_detect)
        #     in_max_east = np.max(_in_easts_detect)
        #     in_min_north = np.min(_in_norths_detect)
        #     in_min_east = np.min(_in_easts_detect)

        #     north_avg_max = (os_max_north + in_max_north) / 2.0
        #     north_avg_min = (os_min_north + in_min_north) / 2.0
        #     east_avg_max = (os_max_east + in_max_east) / 2.0
        #     east_avg_min = (os_min_east + in_min_east) / 2.0

        #     north_span = north_avg_max - north_avg_min
        #     east_span = east_avg_max - east_avg_min

        #     # north_mid = (north_avg_max + north_avg_min) / 2.0
        #     # east_mid = (east_avg_max + east_avg_min) / 2.0

        #     # north_lim = (north_mid - north_span, north_mid + north_span)
        #     # east_lim = (east_mid - east_span, east_mid + east_span)
        # else:
        #     north_span = os_max_north - os_min_north
        #     east_span = os_max_east - os_min_east

        #     # north_mid = (os_max_north + os_min_north) / 2.0
        #     # east_mid = (os_max_east + os_min_east) / 2.0

        #     # north_lim = (north_mid - north_span, north_mid + north_span)
        #     # east_lim = (east_mid - east_span, east_mid + east_span)

        # # map_xlim = (encounter_centre[0] - half_size, encounter_centre[0] + half_size)
        # # map_ylim = (encounter_centre[1] - half_size, encounter_centre[1] + half_size)

        # map_xlim = (encounter_centre[0] - east_span, encounter_centre[0] + east_span)
        # map_ylim = (encounter_centre[1] - north_span, encounter_centre[1] + north_span)

        plt.figure()
        plt.plot(_closest_easts, _closest_norths, color='m') # closest approach
        plt.annotate(f"\n\n    {self._hmd[self._closest_i]:.0f}m", encounter_centre_tup, ha="left", va="center", fontsize=6)
        plt.plot(_os_easts, _os_norths, color='b') # ownship
        # plt.plot(_in_easts, _in_norths, color='r') # intruder
        if daa:
            plt.plot(_in_easts_detect, _in_norths_detect, color='r', alpha=0.2)
            plt.plot(_in_easts_sm, _in_norths_sm, color='r')
        plt.scatter(_os_easts[0], _os_norths[0], color='g', marker='o') # ownship start
        plt.scatter(_in_easts[0], _in_norths[0], color='g', marker='o') # intruder start
        plt.plot(self.geo.ownship_east_m, self.geo.ownship_north_m, color='k', linestyle='dotted', alpha=0.5) # planned ownship
        plt.plot(self.geo.intruder_east_m, self.geo.intruder_north_m, color='k', linestyle='dotted', alpha=0.5) # planned intruder
        # if len(self._avoids) > 0:
        #     _avoid_norths, _avoid_easts, _avoid_alts = zip(*self._avoids)
        #     plt.scatter(_avoid_easts, _avoid_norths, color='y') # avoid guidance
        # plt.scatter(_rejoin_easts, _rejoin_norths, color='c') # avoid guidance
        # plt.quiver(intruder_east_m[0::100], intruder_north_m[0::100], intruder_u_vec[0::100], intruder_v_vec[0::100], color='r')
        # plt.quiver(_os_easts[0::20], _os_norths[0::20], ownship_u_vec[0::20], ownship_v_vec[0::20], color='b')
        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        plt.title(daa_string)
        plt.axis('equal')
        # plt.xlim(map_xlim)
        # plt.ylim(map_ylim)
        plt.savefig(self.results_home + ("{}[{}]_map_out.png".format(daa_string, self.encounter_idx)),    # filename (extension sets format)
                    dpi=300,                # resolution in dots-per-inch
                    bbox_inches="tight",    # trim whitespace
                    transparent=False       # make background transparent
        )
        
        plt.close('all') # close all plots to save memory
        
        # Miss Score
        miss_score = (self._tcpa, self._md[self._closest_i], self._hmd[self._closest_i], self._vmd[self._closest_i])
        lwc_score = ()
        if len(self._lwc_is) > 0:
            lwc_score = (self._lwc_hmd, self._lwc_vmd, self._times[self._lwc_is[0]], self._times[self._lwc_is[-1]])
        else:
            lwc_score = (float("NaN"), float("NaN"), float("NaN"), float("NaN"))
        
        nmac_score = ()
        if len(self._nmac_is) > 0:
            nmac_score = (self._nmac_hmd, self._nmac_vmd, self._times[self._nmac_is[0]], self._times[self._nmac_is[-1]])
        else:
            nmac_score = (float("NaN"), float("NaN"), float("NaN"), float("NaN"))

        # Daidalus Guidance
        if not len(self._first_guidance) > 0:
            self._first_guidance = [float("NaN"), "no guidance"]
        
        # Avg dt
        dt_avg = np.average(np.diff(self._times))
        

        

        
        # - csv gen -
        path_dict = {
            "time": self._times,
            "os_norths (m)": _os_norths,
            "os_easts (m)": _os_easts,
            "os_alts (m)": _os_alts,
            "os_hdg (rad)": self._os_hdg_rad,
            "os_gs (m/s)": self._os_spd,
            "os_vs (m/s)": self._os_vspd,
            "in_norths (m)": _in_norths,
            "in_easts (m)": _in_easts,
            "in_alts (m)": _in_alts,
            "in_hdg (rad)": self._in_hdg_rad,
            "in_gs (m/s)": self._in_spd,
            "in_vs (m/s)": self._in_vspd,
            "pitch (rad)": self._pitch_rad,
            "roll (rad)": self._roll_rad,
            "yaw (rad)": self._yaw_rad,
            "pitch rate (rad)": self._pitch_rate_rad,
            "roll rate (rad)": self._roll_rate_rad,
            "yaw rate (rad)": self._yaw_rate_rad,
            "hmd (m)": self._hmd,
            "vmd (m)": self._vmd,
            "had lwc": self._lwcs,
            "had nmac": self._nmacs,
            "guidance count": self._guidance_counts,
            "last guidance": self._last_guidances,
            "pitch servo": self._pitch_outs,
            "roll servo": self._roll_outs,
            "yaw servo": self._yaw_outs,
            "daidalus ownship input": self._daidalus_os,
            "daidalus intruder input": self._daidalus_in,
            "daidalus reponse": self._daidalus_response,
            "daidalus time": self._daidalus_times,
            "detections": self._detections,
            "tracks": self._tracks,
        }

        self.fill_empty(path_dict, len(self._times))

        # Detect and Track logs
        # detect_length = len(self._detections) # hopefully the same as track length
        # for i in range(detect_length):
        #     det = self._detections[i]
        #     trk = self._tracks[i]

        #     # Detect
        #     for camera in det.get('detections'):
        #         camera_detections = {}
        #         for detection in camera:
        #             detectionLog = {
        #                 "cx": detection.cx,
        #                 "cy": detection.cy,
        #                 "distance": detection.distance,
        #                 "confidence": detection.confidence,
        #                 "dx": detection.dx,
        #                 "dy": detection.dy,
        #                 "above_horizon": detection.above_horizon,
        #                 "cam_id": detection.cam_id
        #             }

        #     # Track
        #     for track_id in trk.keys():
        #         track = trk.get(track_id)
        #         for tkey in track:
        #             path_dict[f"cam {track_id} {tkey}"] = track.get(tkey)
        #             print("adding new key= " + str(f"cam {track_id} {tkey}"))

        # Write to CSV
        output_file = self.results_home + ("{}[{}]_out.csv".format(daa_string, self.encounter_idx))
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(path_dict.keys())
            # Write rows by zipping the values column-wise
            for row in zip(*path_dict.values()):
                writer.writerow(row)
                
        
        return err_score, miss_score, self._skip_count, dt_avg, lwc_score, nmac_score, self._first_guidance