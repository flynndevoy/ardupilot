from Tracker_and_SUM.daidalusClient import DaidalusClient
import scipy
import numpy as np
import time
import navpy
from Tracker_and_SUM.airspaceStatistics import AirspaceStatistics
from Tracker_and_SUM.sixDOFSim import SixDOFSim
from Tracker_and_SUM.daaSystemModel import DAASystemModel
import mat73
import json
import pandas as pd
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
from pathlib import Path
import pickle
import inspect

""" # Add the folder 2 levels up to the path
parent_2 = Path(__file__).resolve().parents[2]
sys.path.append(str(parent_2)) """
 
# Now you can import
from encounter_modelling_classes import *
from sum_functions import *

class Main():
    
    def __init__(self) -> None:
        self.TARGET_LOOP_UPDATE_FREQ = 100000000 #hz
        self.target_loop_wait_time = 1.0/self.TARGET_LOOP_UPDATE_FREQ
        
        self.dt = 0.1 # actual timestep in matlab file

        self.intruder_track_id = 1 # start at 1

        self.statistics = AirspaceStatistics()

        self.use_daidalus = True
        self.use_daa_system_model = True

        if self.use_daa_system_model:
            self.daa_model = DAASystemModel()

    def save_and_plot_encounter(self, encounter_id, ownship_lla, intruder_lla_gt, intruder_lla_raw, intruder_lla_smoothed, track_ids, ref_lla, first_guidance_idx):
        # Convert from LLA to NED (North, East, Down)
        # ownship_ned = navpy.lla2ned((ownship_lla[:,0]), (ownship_lla[:,1]), (ownship_lla[:,2]), (ref_lla[0]), (ref_lla[1]), (ref_lla[2]))
        # intruder_ned = navpy.lla2ned((intruder_lla[:,0]), (intruder_lla[:,1]), (intruder_lla[:,2]), (ref_lla[0]), (ref_lla[1]), (ref_lla[2]))

        # # Prepare DataFrame to save data
        # delta_east = intruder_ned[:,1] - ownship_ned[:,1]
        # delta_north = intruder_ned[:,0] - ownship_ned[:,0]
        # hmd = np.sqrt(delta_east**2 + delta_north**2)
        # df = pd.DataFrame({
        #     'ownship_north_m': ownship_ned[:,0],
        #     'ownship_east_m': ownship_ned[:,1],
        #     'ownship_up_m': ownship_ned[:,2],  # Convert Down to Up
        #     'intruder_north_m': intruder_ned[:,0],
        #     'intruder_east_m': intruder_ned[:,1],
        #     'intruder_up_m': intruder_ned[:,2],   # Convert Down to Up,
        #     'intruder_ground_speed_mps': intruder_lla[:,3],  # Ground speed in m/s
        #     'ownship_ground_speed_mps': ownship_lla[:,3],  # Ground speed in m/s
        #     'intruder_vertical_speed_mps': intruder_lla[:,4],  # Vertical speed in m/s
        #     'ownship_vertical_speed_mps': ownship_lla[:,4],  # Vertical speed in m/s
        #     'hmd_m': hmd,  # Horizontal Miss Distance
        #     'vmd_m': intruder_ned[:, 2] - ownship_ned[:, 2],  # Vertical Miss Distance
        #     'time_sec': ownship_lla[:,5]  # Time in seconds
        # })

        # if first_guidance_idx is None:
        #     df['guidance_received'] = 'No Guidance Received'
        # else:
        #     df['guidance_received'] = first_guidance_idx
            

        # Save data to CSV
        # csv_filename = f"results/Encounter_CSV/{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')}_encounter_{encounter_id}.csv"
        # df.to_csv(csv_filename, index=False)
        # print(f'Data saved to {csv_filename}')

        # # === 2D North vs East Plot ===
        # plt.figure(figsize=(8, 6))

        # # Full track lines
        # plt.plot(ownship_lla[:, 2], label='Ownship Alt')
        # plt.plot(intruder_lla[:, 2], label='Intruder Alt')

        # plt.xlabel('East (m)')
        # plt.ylabel('North (m)')
        # plt.title(f'Encounter {encounter_id}: North vs East')
        # plt.legend()
        # plt.grid(True)
        # plt.axis('equal')

        # plt.show()
        # === 2D North vs East Plot ===
        plt.figure(figsize=(8, 6))

        # Full track lines
        plt.plot(ownship_lla[:, 1], ownship_lla[:, 0], label='Ownship Track')
        plt.plot(intruder_lla_gt[:, 1], intruder_lla_gt[:, 0], color='gray', label='Intruder Track Ground Truth')
        plt.plot(intruder_lla_raw[:, 1], intruder_lla_raw[:, 0], color='gray', label='Intruder Track Raw', alpha=0.3)
        # plt.plot(intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 0], label='Intruder Track Smoothed')

        track_ids_unique = np.unique(track_ids)  # Get unique track IDs
        cmap = plt.get_cmap('tab10')  # Or use 'tab20', 'hsv', etc.

        for i, track_id in enumerate(track_ids_unique):
            mask = track_ids == track_id
            
            north = intruder_lla_smoothed[mask, 0]
            east = intruder_lla_smoothed[mask, 1]
            vel_north = intruder_lla_smoothed[mask, 3]
            vel_east = intruder_lla_smoothed[mask, 4]

            plt.plot(
                east, north,
                color=cmap(i % cmap.N),
                label=f'Smoothed Track ID: {track_id}'
            )

            # Add velocity vectors using quiver
            plt.quiver(
                east, north,       # Positions
                vel_east, vel_north,   # Components of velocity (U=East, V=North)
                color=cmap(i % cmap.N),
                angles='xy',
                scale_units='xy',
                scale=1,
                width=0.0025,
                alpha=0.7
            )

        # Start points
        plt.scatter(ownship_lla[0, 1], ownship_lla[0, 0], color='green', marker='o', s= 80, label='Ownship Start')
        # plt.scatter(intruder_lla_gt[0, 1], intruder_lla_gt[0, 0], color='green', marker='o', s= 80, label='Intruder Start')
        # plt.scatter(intruder_lla_orig[0, 1], intruder_lla_orig[0, 0], color='green', marker='o', s= 80, label='Ideal Intruder Start')
        plt.scatter(intruder_lla_smoothed[0, 1], intruder_lla_smoothed[0, 0], color='green', marker='o', s= 80, label='Smoothed Intruder Start')

        # End points
        plt.scatter(ownship_lla[-1, 1], ownship_lla[-1, 0], color='red', marker='o', s= 80,  label='Ownship End')
        # plt.scatter(intruder_lla_gt[-1, 1], intruder_lla_gt[-1, 0], color='red', marker='o', s= 80,  label='Intruder End')
        # plt.scatter(intruder_lla_orig[-1, 1], intruder_lla_orig[-1, 0], color='red', marker='o', s= 80,  label='Ideal Intruder End')
        plt.scatter(intruder_lla_smoothed[-1, 1], intruder_lla_smoothed[-1, 0], color='red', marker='o', s= 80,  label='Smoothed Intruder End')

        plt.xlabel('East (m)')
        plt.ylabel('North (m)')
        filename = os.path.basename(__file__) if '__file__' in globals() else 'interactive_session'
        plt.title(f'Encounter {encounter_id}: North vs East ({filename}), ({self.coord_frame})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        plt.show()

        # # === Separation Distance Plots ===
        # plt.figure(figsize=(8, 6))

        # # Calculate horizontal separation (hypotenuse of N/E diff)
        # horizontal_sep = np.sqrt((ownship_lla[:, 0] - intruder_lla[:, 0])**2 +
        #                         (ownship_lla[:, 1] - intruder_lla[:, 1])**2)

        # # Calculate vertical separation (altitude difference)
        # vertical_sep = intruder_lla[:, 2] - ownship_lla[:, 2]

        # # Primary axis (left): Horizontal separation
        # ax1 = plt.gca()
        # ax1.plot(horizontal_sep, label='Horizontal Separation', color='blue')
        # ax1.set_xlabel('Frame')
        # ax1.set_ylabel('Horizontal Separation (meters)', color='blue')
        # ax1.tick_params(axis='y', labelcolor='blue')

        # # Secondary axis (right): Vertical separation
        # ax2 = ax1.twinx()
        # ax2.plot(vertical_sep, label='Vertical Separation', color='red')
        # ax2.set_ylabel('Vertical Separation (meters)', color='red')
        # ax2.tick_params(axis='y', labelcolor='red')

        # # Title and grid
        # plt.title(f'Encounter {encounter_id}: Horizontal vs Vertical Separation')
        # ax1.grid(True)

        # # Optional: Legends (one for each axis)
        # lines1, labels1 = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # plt.tight_layout()

        # # === 3D East-North-Altitude Plot ===
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Full track lines
        # ax.plot(ownship_lla[:, 0], ownship_lla[:, 1], ownship_lla[:, 2], label='Ownship Track')
        # ax.plot(intruder_lla[:, 0], intruder_lla[:, 1], intruder_lla[:, 2], label='Intruder Track')
        # ax.plot(intruder_lla_orig[:, 0], intruder_lla_orig[:, 1], intruder_lla_orig[:, 2], label='Intruder Track Ideal')
        # ax.plot(intruder_lla_smoothed[:, 0], intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 2], label='Intruder Track Smoothed')

        # # Start points
        # ax.scatter(ownship_lla[0, 0], ownship_lla[0, 1], ownship_lla[0, 2], color='green', marker='o', s= 80, label='Ownship Start')
        # ax.scatter(intruder_lla[0, 0], intruder_lla[0, 1], intruder_lla[0, 2], color='green', marker='o', s= 80, label='Intruder Start')
        # ax.scatter(intruder_lla[0, 0], intruder_lla[0, 1], intruder_lla[0, 2], color='green', marker='o', s= 80, label='Intruder Start')

        # # End points
        # ax.scatter(ownship_lla[-1, 0], ownship_lla[-1, 1], ownship_lla[-1, 2], color='red', marker='o', s= 80, label='Ownship End')
        # ax.scatter(intruder_lla[-1, 0], intruder_lla[-1, 1], intruder_lla[-1, 2], color='red', marker='o', s= 80, label='Intruder End')

        # ax.set_xlabel('East (m)')
        # ax.set_ylabel('North (m)')
        # ax.set_zlabel('Altitude (m)')
        # ax.set_title(f'Encounter {encounter_id}: 3D Track (East-North-Altitude)')
        # ax.legend()
        # ax.grid(True)

        # # === Force equal aspect ratio ===
        # all_points = np.vstack((ownship_lla, intruder_lla))
        # x_range = all_points[:, 0]
        # y_range = all_points[:, 1]
        # z_range = all_points[:, 2]

        # x_min, x_max = np.nanmin(x_range), np.nanmax(x_range)
        # y_min, y_max = np.nanmin(y_range), np.nanmax(y_range)
        # z_min, z_max = np.nanmin(z_range), np.nanmax(z_range)

        # max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

        # mid_x = (x_max + x_min) / 2.0
        # mid_y = (y_max + y_min) / 2.0
        # mid_z = (z_max + z_min) / 2.0

        # ax.set_xlim(mid_x - max_range, mid_x + max_range)
        # ax.set_ylim(mid_y - max_range, mid_y + max_range)
        # ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.show()


    def load_mat_file(self, fp):
        # Load the .mat file
        try:
            mat_contents = scipy.io.loadmat(fp)
            use_scipy = True
            return mat_contents, use_scipy
        except NotImplementedError as e:
            print(f"Could not load MATLAB file with scipy: {e}. \nTrying again with mat73.")
            mat_contents = mat73.loadmat(fp)
            use_scipy = False
            return mat_contents, use_scipy
        
    def get_mat_param(self, mat, field, idx, param, use_scipy):
        if use_scipy:
            return mat[field][idx][param]
        else:
            return mat[field][param][idx]
        
    def save_summary_to_disk(self, summary):
        try:
            # Save dictionary to a JSON file
            file_path = f"results/summary_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H')}.json"
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, "w") as file:
                json.dump(summary, file, indent=4)  # 'indent=4' is used to format the JSON nicely
                print(f"Log saved to: {file_path}")
        except Exception as e:
            print(f"Error saving log to disk: {e}")


    def intruder_rel_x_y_z(self, ownship_packet, intruder, heading_ref_deg = 0):
        # Ownship absolute position in NED (North, East, Up)
        ref_n = ownship_packet["north_m"]
        ref_e = ownship_packet["east_m"]
        ref_u = ownship_packet["alt"]
     
        # Intruder absolute position in NED
        i_n = intruder["north_m"]
        i_e = intruder["east_m"]
        i_u = intruder["alt"]
     
        # Relative position (from ownship to intruder) in NED frame
        delta_n = i_n - ref_n
        delta_e = i_e - ref_e
        delta_u = i_u - ref_u  # Up component
     
        # Ownship heading (track angle) in degrees
        heading_deg = heading_ref_deg
        heading_rad = np.radians(heading_deg)
     
        # Rotate NED delta into ownship body frame (X = forward/North, Y = right/East, Z = down)
        x = delta_n * np.cos(heading_rad) + delta_e * np.sin(heading_rad)
        y = -delta_n * np.sin(heading_rad) + delta_e * np.cos(heading_rad)
        z = delta_u
     
        # Intruder track angle relative to ownship heading
        i_trk = intruder["trk"]
        rel_trk = (i_trk - heading_deg + 360) % 360
     
        return {
            "x": x,    # Forward (meters)
            "y": y,    # Right (meters)
            "z": z,    # Down (meters)
            "t": rel_trk  # Relative track angle in degrees (body frame, 0° = forward, positive clockwise)
        }

    def run(self,ownship_file, intruder_file,selected_encounters=None):

        #ownship_file_list = ['C:/Users/aries/Downloads/Rev_Aero_Stuff/Daidalus/Encounters/Jabiru Based Encounters/70s TCPA/1000os_data_at_TCPA_70.mat']
        
        #intruder_file_list = ['C:/Users/aries/Downloads/Rev_Aero_Stuff/Daidalus/Encounters/Jabiru Based Encounters/70s TCPA/1000intruder_data_at_TCPA_70.mat']

        self.intruder_track_id = 1

        for ownship_file, intruder_file in [(ownship_file, intruder_file)]:
            
            ownship_mat_data, use_scipy_ownship = self.load_mat_file(ownship_file)
            intruder_mat_data, use_scipy_intruder = self.load_mat_file(intruder_file)

            self.client = DaidalusClient('localhost', 65432)

            with self.client.connect_to_daidalus_server() as connection:

                if use_scipy_ownship:
                    num_encounters = len(ownship_mat_data['ownship_data'])-1
                else:
                    num_encounters = len(ownship_mat_data['ownship_data']['time'])-1

                if selected_encounters is None:
                    selected_encounters = range(num_encounters)
                else:
                    selected_encounters = [idx for idx in selected_encounters if idx < num_encounters]

                guidance_info = {}
                

                for encounter_idx in tqdm(selected_encounters, desc="Encounters"):

                    self.ownship.reset('OS' + str(encounter_idx))
                    enc_this = Encounter(self.ownship)
                    tracker = Tracker(self.ownship)
                    self.encounters[encounter_idx] = enc_this
 
                    self.start_simulating_ownship = False
                    self.waited_for_pilot_reponse = False
                    self.ownship_sim = None
                    start_sim_idx = None

                    ### Coordinate Frame Preferences
                    cartesian = True
                    coord_frame = "LLA"
                    if cartesian:
                        coord_frame = "NEU"
                        centered = False
                        aligned = False

                        if aligned and not centered:
                            print("Must center if aligning!")
                            error_out_here
                    else:
                        centered = False
                        aligned = False

                    if cartesian:
                        if aligned:
                            self.coord_frame = "NEU_aligned"
                        elif centered:
                            self.coord_frame = "NEU_centered"
                        else:
                            self.coord_frame = "NEU"
                    else:
                        self.coord_frame = "LLA"

                    ownship_lla_list = []
                    intruder_lla_list_gt = []
                    intruder_lla_list_raw = []
                    intruder_lla_list_smoothed = []
                    track_ids = []
                    guidance = []
                    input_coords = []

                    ownship_time_sec = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'time', use_scipy_ownship)[0])
                    ownship_east_m = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'east_ft', use_scipy_ownship)[0]) * 0.3048
                    ownship_north_m = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'north_ft', use_scipy_ownship)[0]) * 0.3048
                    ownship_up_m = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'up_ft', use_scipy_ownship)[0]) * 0.3048
                    ownship_hdg_rad = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'psi_rad', use_scipy_ownship)[0])
                    ownship_phi_rad = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'phi_rad', use_scipy_ownship)[0])
                    ownship_theta_rad = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'theta_rad', use_scipy_ownship)[0])
                    ownship_speed_m_per_s = np.squeeze(self.get_mat_param(ownship_mat_data, 'ownship_data', encounter_idx, 'speed_ftps', use_scipy_ownship)[0]) * 0.3048
                    ownship_dz = np.diff(ownship_up_m)
                    ownship_vz = ownship_dz / self.dt

                    intruder_east_m = np.squeeze(self.get_mat_param(intruder_mat_data, 'intruder_data', encounter_idx, 'east_ft', use_scipy_intruder)[0]) * 0.3048
                    intruder_north_m = np.squeeze(self.get_mat_param(intruder_mat_data, 'intruder_data', encounter_idx, 'north_ft', use_scipy_intruder)[0]) * 0.3048
                    intruder_up_m = np.squeeze(self.get_mat_param(intruder_mat_data, 'intruder_data', encounter_idx, 'up_ft', use_scipy_intruder)[0]) * 0.3048
                    intruder_hdg_rad = np.squeeze(self.get_mat_param(intruder_mat_data, 'intruder_data', encounter_idx, 'psi_rad', use_scipy_intruder)[0])
                    intruder_speed_m_per_s = np.squeeze(self.get_mat_param(intruder_mat_data, 'intruder_data', encounter_idx, 'speed_ftps', use_scipy_intruder)[0]) * 0.3048
                    # Calculate the difference in altitude and time
                    intruder_dz = np.diff(intruder_up_m)  # Change in altitude
                    intruder_vz = intruder_dz / self.dt  # Vertical velocity (rate of altitude change)

                    # Define reference point (origin) in LLA (latitude, longitude, altitude)
                    self.ref_lla = (-27.3833318, 152.499998, 0) #coominya for example

                    first_guidance_idx = None
                    home_lla_recorded = False
                    sim_start = time.time()

                    for row_idx in range(len(ownship_east_m)-1):

                        start = time.time()

                        ownship_heading_deg = float(np.degrees(ownship_hdg_rad[row_idx]))
                        ownship_heading_deg = (ownship_heading_deg + 360) % 360
                        # ownship_vx = ownship_speed_m_per_s[idx] * np.cos(ownship_hdg_rad[idx])
                        # ownship_vy = ownship_speed_m_per_s[idx] * np.sin(ownship_hdg_rad[idx])
                        
                        # Generate initial ownship LLA Coords 
                        ownship_lla = navpy.ned2lla([float(ownship_north_m[row_idx]), float(ownship_east_m[row_idx]), float(-ownship_up_m[row_idx])], self.ref_lla[0], self.ref_lla[1], 0)
                        ownship_ver_vel = float(ownship_vz[row_idx])
                        ownship_hor_vel = float(ownship_speed_m_per_s[row_idx])
                        ownship_time = ownship_time_sec[row_idx]
                        ownship_phi = float(np.degrees(ownship_phi_rad[row_idx]))
                        ownship_theta = float(np.degrees(ownship_theta_rad[row_idx]))
                        ownship_psi = float(np.degrees(ownship_hdg_rad[row_idx]))

                        if not home_lla_recorded:
                            home_lla = ownship_lla
                            home_lla_recorded = True

                        if self.start_simulating_ownship:
                            self.ownship_sim.simulate()
                            
                            # Populate ownship lat lon with the outputted LLA Coords which are based on ref_lla
                            ownship_state = {"lat": self.ownship_sim.lat,
                                            "lon": self.ownship_sim.long,
                                            "alt": self.ownship_sim.alt,
                                            "trk": self.ownship_sim.track,
                                            "rl": 0.0, # SUM (Roll)
                                            "pt": 0.0, # SUM (Pitch)
                                            "yw": 0.0, # SUM (Yaw)
                                            "ground_speed": self.ownship_sim.speed,
                                            "vertical_speed": self.ownship_sim.vert_speed,
                                            "time": ownship_time,
                                            "coord_frame": coord_frame,
                                            "centered": centered,
                                            "aligned": aligned
                            }
                        else:
                            # Populate ownship lat lon with the original LLA Coords using ref_lla
                            ownship_state = {"lat": ownship_lla[0],
                                            "lon": ownship_lla[1],
                                            "alt": ownship_up_m[row_idx],
                                            "trk": ownship_heading_deg,
                                            "rl": 0.0, # SUM (Roll)
                                            "pt": 0.0, # SUM (Pitch)
                                            "yw": 0.0, # SUM (Yaw)
                                            "ground_speed": ownship_hor_vel,
                                            "vertical_speed": ownship_ver_vel,
                                            "time": ownship_time,
                                            "coord_frame": coord_frame,
                                            "centered": centered,
                                            "aligned": aligned
                            }
                            
                        # - Intruder State -
                        intruder_states = []
                        intruder_heading_deg = float(np.degrees(intruder_hdg_rad[row_idx]))
                        intruder_heading_deg = (intruder_heading_deg + 360) % 360

                        # Generate initial intruder LLA Coords based on the same ref_lla that ownship lla coords are based on                        
                        intruder_lla = navpy.ned2lla([float(intruder_north_m[row_idx]), float(intruder_east_m[row_idx]), float(-intruder_up_m[row_idx])], self.ref_lla[0], self.ref_lla[1], 0)
                        intruder_lla = (intruder_lla[0], intruder_lla[1], intruder_up_m[row_idx])
                        intruder_ver_vel = float(intruder_vz[row_idx])
                        intruder_hor_vel = float(intruder_speed_m_per_s[row_idx])

                        intruder_state = {"lat": intruder_lla[0],
                                        "lon": intruder_lla[1],
                                        "alt": intruder_up_m[row_idx],
                                        "trk": intruder_heading_deg,
                                        "ground_speed": intruder_hor_vel,
                                        "vertical_speed": intruder_ver_vel,
                                        "type": "DAA",
                                        "callsign": "INTRUDER",
                                        "id": str(encounter_idx),
                                        "time": ownship_time,
                                        "coord_frame": coord_frame,
                                        "centered": centered,
                                        "aligned": aligned
                        }

                        ### SUM

                        ##### Now generate ownship and intruder cartesian coordinates relative to the home_lla (aka ref_lla is not used past this point) #####

                        # Build ownship north-east cartesian coords relative to home_lla
                        os_neu = navpy.lla2ned(
                            ownship_state['lat'],
                            ownship_state['lon'],
                            -ownship_state['alt'],
                            home_lla[0],
                            home_lla[1],
                            0
                        )

                        self.ownship.set_state(
                            (os_neu[0], os_neu[1], ownship_state["alt"]), 
                            (ownship_state["rl"], ownship_state["pt"], ownship_state["trk"] + ownship_state["yw"])
                            )

                        ownship_state["north_m"] = os_neu[0]
                        ownship_state["east_m"] = os_neu[1]

                        # Build intruder north-east cartesian coords relative to home_lla
                        int_neu = navpy.lla2ned(
                                intruder_state['lat'],
                                intruder_state['lon'],
                                -intruder_state['alt'],
                                home_lla[0],
                                home_lla[1],
                                0
                            )

                        intruder_state["north_m"] = int_neu[0]
                        intruder_state["east_m"] = int_neu[1]

                        # Apply relevant coordinate transforms
                        if cartesian: 
                            if centered:
                                ownship_state["north_m_centred"] = 0.0
                                ownship_state["east_m_centred"] = 0.0

                        ###### Process Intruder and Ownship Input State #####

                        intx, inty, intz = intruder_state["north_m"], intruder_state["east_m"], intruder_state["alt"]

                        if SUM_ON:
                            intruder_class_list = []
                            intruder_class_list.append(Intruder(idx=1, position_xyz=(intx, inty, intz)))
                            enc_this.update_encounter(row_idx, self.ownship, intruder_class_list)
                            AFMats = enc_this.generate_affine_transform_matrices(plot = False)
                            detections = enc_this.detections_this_frame(ideal=False)
                            new_track_points = tracker.update_tracks(ownship_time, AFMats, detections)
                            new_tracks_detected = len(new_track_points) > 0

                            if new_tracks_detected:
                                x_new_raw, y_new_raw, z_new_raw = list(new_track_points.values())[0]['abs_cartesian_x_y_z']
                                x_new_smoothed, y_new_smoothed, z_new_smoothed = list(new_track_points.values())[0]['abs_cartesian_x_y_z_smoothed']
                                grnd_speed_smoothed = np.linalg.norm(list(new_track_points.values())[0]['velocity_x_y_z_smoothed'][:2])
                                ver_speed_smoothed = list(new_track_points.values())[0]['velocity_x_y_z_smoothed'][2]
                                vel_smoothed = list(new_track_points.values())[0]['velocity_x_y_z_smoothed']
                                trk_smoothed = list(new_track_points.values())[0]['trk_smoothed']
                                trk_id = list(new_track_points.keys())[0]
                                intruder_state["range_uncertainty"] = list(new_track_points.values())[0]['distance_variance']
                            else:
                                x_new_raw, y_new_raw, z_new_raw = np.nan, np.nan, np.nan
                                x_new_smoothed, y_new_smoothed, z_new_smoothed = np.nan, np.nan, np.nan
                                vel_smoothed = np.array(np.nan, np.nan, np.nan)
                        else:
                            new_tracks_detected = True
                            x_new_raw, y_new_raw, z_new_raw = intx, inty, intz
                            x_new_smoothed, y_new_smoothed, z_new_smoothed = intx, inty, intz
                            grnd_speed_smoothed = intruder_state["ground_speed"]
                            ver_speed_smoothed = intruder_state["vertical_speed"]
                            trk_smoothed = intruder_state["trk"]
                            vel_smoothed = [grnd_speed_smoothed * np.sin(trk_smoothed),
                                            grnd_speed_smoothed * np.cos(trk_smoothed),
                                            ver_speed_smoothed]
                            intruder_state["range_uncertainty"] = 0
                            trk_id = 1

                        # Apply relevant coordinate transforms
                        if new_tracks_detected:
                            # Overwrite intruder state with noisy values
                            intruder_state["north_m"] = x_new_smoothed
                            intruder_state["east_m"] = y_new_smoothed
                            intruder_state["alt"] = z_new_smoothed
                            intruder_state["trk"] = trk_smoothed
                            intruder_state["ground_speed"] = grnd_speed_smoothed
                            intruder_state["vertical_speed"] = ver_speed_smoothed

                            intruder_state["lat"], intruder_state["lon"], _ = navpy.ned2lla(
                                [intruder_state["north_m"], intruder_state["east_m"], intruder_state["alt"]],
                                home_lla[0], home_lla[1], 0
                                )

                            if cartesian:   
                                if centered and not aligned:
                                    intruder_rel = self.intruder_rel_x_y_z(ownship_state, intruder_state, heading_ref_deg = 0)

                                    intruder_state["north_m_centred"] = intruder_rel['x']
                                    intruder_state["east_m_centred"] = intruder_rel['y']
        
                                if centered and aligned:
                                    intruder_rel = self.intruder_rel_x_y_z(ownship_state, intruder_state, heading_ref_deg = ownship_state["trk"])

                                    intruder_state["north_m_centred"] = intruder_rel['x']
                                    intruder_state["east_m_centred"] = intruder_rel['y']
                                    intruder_state["trk_aligned"] = intruder_rel['t']

                            intruder_states.append(intruder_state)                

                        ownship_lla_list.append([ownship_state["north_m"], ownship_state["east_m"], ownship_state["alt"], ownship_state["ground_speed"], ownship_state["vertical_speed"], ownship_state["time"]])
                        intruder_lla_list_gt.append([intx, inty, intz, intruder_state["ground_speed"], intruder_state["vertical_speed"], intruder_state["time"]])
                        targets_updated = False
                        if self.use_daidalus and new_tracks_detected:
                            # get daidalus output from server
                            # network_start =  time.time()
                            intruder_lla_list_raw.append([x_new_raw, y_new_raw, z_new_raw, intruder_state["ground_speed"], intruder_state["vertical_speed"], intruder_state["time"]])
                            intruder_lla_list_smoothed.append([x_new_smoothed, y_new_smoothed, z_new_smoothed, *vel_smoothed])
                            track_ids.append(trk_id)
                            
                            response = self.client.send_data(connection, ownship_state, intruder_states)

                            input_coords.append([ownship_time, self.intruder_rel_x_y_z(ownship_state, intruder_states[0])])
                            # print(f"Daidalus network took: {(time.time() - network_start)*1000} ms")

                            if np.isfinite(response['suggested_hdir_deg']) or np.isfinite(response['suggested_hspd_knot']) or np.isfinite(response['suggested_vspd_fpm']) or np.isfinite(response['suggested_alt_ft']):
                                if first_guidance_idx is None:
                                    first_guidance_idx = row_idx
                                    # print(first_guidance_idx)
                                    # print({'o': ownship_state, 'd': intruder_states})
                                    # print(response)
                                    # cow
                                    # dx = float(ownship_east_m[row_idx]) - float(intruder_east_m[row_idx])
                                    # dy = float(ownship_north_m[row_idx]) - float(intruder_north_m[row_idx])
                                    # dz = float(ownship_up_m[row_idx]) - float(intruder_up_m[row_idx])

                                    # hmd = float(np.sqrt(dx**2 + dy**2))
                                    # vmd = float(abs(dz))
                                    # guidance_info[f'encounter_{encounter_idx}'] = {
                                    #     'first_guidance_idx': first_guidance_idx,
                                    #     'ownship_time': ownship_time,
                                    #     'suggested_hdir_deg': response['suggested_hdir_deg'],
                                    #     'suggested_hspd_knot': response['suggested_hspd_knot'],
                                    #     'suggested_vspd_fpm': response['suggested_vspd_fpm'],
                                    #     'suggested_alt_ft': response['suggested_alt_ft'],
                                    #     'ownship_data': ownship_state,
                                    #     'intruder_data': intruder_state,
                                    #     'hmd_m': hmd,
                                    #     'vmd_m': vmd,
                                    #     'dx_m': dx,
                                    #     'dy_m': dy,
                                    #     'ownship_up_m': ownship_up_m[row_idx],
                                    #     'ownship_east_m': ownship_east_m[row_idx],
                                    #     'ownship_north_m': ownship_north_m[row_idx],
                                    #     'intruder_east_m': intruder_east_m[row_idx],
                                    #     'intruder_north_m': intruder_north_m[row_idx],
                                    #     'intruder_up_m': intruder_up_m[row_idx],
                                    # }

                                if response["intruders"]:
                                    if response["intruders"][0]["alert_level"] == 2:
                                        self.statistics.log_alert(intruder_id=intruder_state['id'], alert_type='CA', timestamp=ownship_time)
                                    if response["intruders"][0]["alert_level"] == 3:
                                        self.statistics.log_alert(intruder_id=intruder_state['id'], alert_type='WA', timestamp=ownship_time)

                                suggested_hdir_deg = response['suggested_hdir_deg'] if np.isfinite(response['suggested_hdir_deg']) else ownship_state['trk']
                                suggested_hspd_mps = response['suggested_hspd_knot']/1.944 if np.isfinite(response['suggested_hspd_knot']) else ownship_state['ground_speed']
                                suggested_vspd_mps = response['suggested_vspd_fpm']/196.9 if np.isfinite(response['suggested_vspd_fpm']) else ownship_state['vertical_speed']
                                suggested_alt_m = response['suggested_alt_ft']/3.281 if np.isfinite(response['suggested_alt_ft']) else ownship_state['alt']

                                start_sim_idx = row_idx

                                if not self.start_simulating_ownship and start_sim_idx == row_idx:
                                    
                                    #print("Taking daidalus guidance and beginning ownship simulation")
                                    self.start_simulating_ownship = True
                                    #psi is yaw, phi is roll, theta is pitch
                                    self.ownship_sim = SixDOFSim(ownship_state['lat'], ownship_state['lon'], ownship_state['alt'], ownship_state['trk'], ownship_state['ground_speed'], ownship_state['vertical_speed'], self.dt)
                                    
                                
                                if self.start_simulating_ownship:
                                    self.ownship_sim.update_targets(suggested_hdir_deg, suggested_hspd_mps, suggested_vspd_mps, suggested_alt_m)
                                    targets_updated = True

                                # print("-------------")
                                # print(f"suggested heading: {suggested_hdir_deg}")
                                # print(f"suggested speed: {suggested_hspd_mps}")
                                # print(f"suggested vert speed: {suggested_vspd_mps}")
                                # print(f"suggested alt: {suggested_alt_m}")
                                # print("-------------")

                        if self.start_simulating_ownship and not targets_updated:
                            self.ownship_sim.update_targets(ownship_state['trk'], ownship_state['ground_speed'], ownship_state['vertical_speed'], ownship_state['alt'])

                        # stats = self.statistics.evaluate(ownship_data, intruder_data_stats)
                        
                        # print(stats)
                        # print(f"Total Loop took: {(time.time() - start)*1000} ms")
                        time.sleep(max(self.target_loop_wait_time - (time.time() - start), 0))
                    
                    # === Extract from input_coords ===
                    # x_vals = [entry[1]['x'] for entry in input_coords]
                    # y_vals = [entry[1]['y'] for entry in input_coords]
                    # z_vals = [entry[1]['z'] for entry in input_coords]
                    # t_vals = [entry[1]['t'] for entry in input_coords]

                    # # === Extract from guidance ===
                    # times = [entry[0] for entry in input_coords]
                    # times2 = [entry['ownship_time'] for entry in guidance]

                    # headings = [val if np.isfinite(val) else np.nan for val in (entry['suggested_hdir_deg'] for entry in guidance)]
                    # speeds = [val if np.isfinite(val) else np.nan for val in (entry['suggested_hspd_knot'] for entry in guidance)]
                    # vert_speeds = [val if np.isfinite(val) else np.nan for val in (entry['suggested_vspd_fpm'] for entry in guidance)]
                    # altitudes = [val if np.isfinite(val) else np.nan for val in (entry['suggested_alt_ft'] for entry in guidance)]

                    # # === Create 4x2 Subplots ===
                    # fig, axs = plt.subplots(4, 2, figsize=(14, 16), sharex=True)

                    # # Left column: input_coords plots
                    # axs[0, 0].plot(times, x_vals, marker='o')
                    # axs[0, 0].set_ylabel('X (Forward, m)')
                    # axs[0, 0].set_title('Intruder Relative X vs Frame')
                    # axs[0, 0].grid(True)

                    # axs[1, 0].plot(times, y_vals, marker='o')
                    # axs[1, 0].set_ylabel('Y (Right, m)')
                    # axs[1, 0].set_title('Intruder Relative Y vs Frame')
                    # axs[1, 0].grid(True)

                    # axs[2, 0].plot(times, z_vals, marker='o')
                    # axs[2, 0].set_ylabel('Z (Down, m)')
                    # axs[2, 0].set_title('Intruder Relative Z vs Frame')
                    # axs[2, 0].grid(True)

                    # axs[3, 0].plot(times, t_vals, marker='o')
                    # axs[3, 0].set_ylabel('Relative Track Angle (deg)')
                    # axs[3, 0].set_xlabel('Frame')
                    # axs[3, 0].set_title('Intruder Relative Track Angle vs Frame')
                    # axs[3, 0].grid(True)

                    # # Right column: guidance plots
                    # axs[0, 1].plot(times2, headings, marker='o')
                    # axs[0, 1].set_ylabel('Heading (deg)')
                    # axs[0, 1].set_title('Suggested Heading vs Time')
                    # axs[0, 1].grid(True)

                    # axs[1, 1].plot(times2, speeds, marker='o')
                    # axs[1, 1].set_ylabel('Speed (knots)')
                    # axs[1, 1].set_title('Suggested Speed vs Time')
                    # axs[1, 1].grid(True)

                    # axs[2, 1].plot(times2, vert_speeds, marker='o')
                    # axs[2, 1].set_ylabel('Vertical Speed (fpm)')
                    # axs[2, 1].set_title('Suggested Vertical Speed vs Time')
                    # axs[2, 1].grid(True)

                    # axs[3, 1].plot(times2, altitudes, marker='o')
                    # axs[3, 1].set_ylabel('Altitude (ft)')
                    # axs[3, 1].set_xlabel('Time (s)')
                    # axs[3, 1].set_title('Suggested Altitude vs Time')
                    # axs[3, 1].grid(True)

                    # # === Layout adjustment ===
                    # plt.tight_layout()
                    # plt.show()
                    flight_stats = self.statistics.finishEncounter()

                    print(flight_stats)

                    ownship_lla_array = np.array(ownship_lla_list)

                    intruder_lla_array_gt = np.array(intruder_lla_list_gt)
                    intruder_lla_array_raw = np.array(intruder_lla_list_raw)
                    intruder_lla_array_smoothed = np.array(intruder_lla_list_smoothed)
                    track_ids = np.array(track_ids)

                    self.save_and_plot_encounter(encounter_idx, ownship_lla_array, intruder_lla_array_gt, intruder_lla_array_raw, intruder_lla_array_smoothed, track_ids, self.ref_lla, first_guidance_idx)
                    
                    self.statistics.reset()
                    self.intruder_track_id = 1

                    self.client.reset_daidalus(connection)

                print(f"Total Stats for {ownship_file}, {intruder_file}:")
                print(f"Daidalus active: {self.use_daidalus}")
                stats = self.statistics.getTotalStats()
                stats['files'] = [ownship_file, intruder_file]
                self.save_summary_to_disk(stats)
                #self.plotter.save_all_encounters(ownship_file.split('/')[-2] + ' complete_encounters')
                print(stats)
                #guidance_info['files'] = [ownship_file, intruder_file]
                output_path = f"results/guidance_info_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')}.json"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                with open(output_path, 'w') as f:
                    json.dump(guidance_info, f, indent=4)

                print(f"Guidance info saved to: {output_path}")

if __name__ == '__main__':
    np.random.seed(1)
    ownship_file = './example_encounters/extracted_ownship_data73.mat'
    intruder_file = './example_encounters/extracted_intruder_data73.mat'

    ###############################################
    ##### Generate Sensor Uncertainty Models ######
    ###############################################
    
    SUM_ON = False
    regenerate_SUM = False
    if SUM_ON:
        SUM_FOLDER = "SUM"
        os.makedirs(SUM_FOLDER, exist_ok=True)
 
        def save_pickle(obj, filename):
            with open(filename, 'wb') as f:
                pickle.dump(obj, f)
 
        def load_pickle(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
 
        def save_numpy(obj, filename):
            np.save(filename, obj)
 
        def load_numpy(filename):
            return np.load(filename, allow_pickle=True)
 
        # Check if saved models exist
        pixel_model_state_path = os.path.join(SUM_FOLDER, "pixel_uncertainty_model.pkl")
        range_bins_path = os.path.join(SUM_FOLDER, "range_bins.npy")
        range_model_input_format_path = os.path.join(SUM_FOLDER, "range_error_model_input_format.pkl")
        range_model_params_path = os.path.join(SUM_FOLDER, "range_model_params.pkl")
        detected_range_model_params_path = os.path.join(SUM_FOLDER, "detected_range_model_params.pkl")
 
        if regenerate_SUM or not (
            os.path.exists(pixel_model_state_path) and
            os.path.exists(range_bins_path) and
            os.path.exists(range_model_input_format_path) and
            os.path.exists(range_model_params_path) and
            os.path.exists(detected_range_model_params_path)
        ):
            print("Generating sensor uncertainty models from data...")
 
            # Load data
            data = import_encounter_data()
 
            # Generate models
            detected_range_uncertainty_model_params = generate_range_uncertainty_model_bins_or_measurement_uncertainty_params(data, bins=30, gt_dist_basis = False)
            pixel_uncertainty_model, pixel_uncertainty_model_state_data = generate_pixel_uncertainty_model(all_detection_summaries = data, bins=20)
            range_bins = generate_range_uncertainty_model_bins_or_measurement_uncertainty_params(data, bins=30, error_trim=2500)
            range_uncertainty_model, range_model_params, range_error_model_input_format = generate_range_uncertainty_model(all_detection_summaries = data, range_bins = range_bins, error_trim=2500)
 
            # === Save state only ===
            save_pickle(pixel_uncertainty_model_state_data, pixel_model_state_path)
            save_numpy(range_bins, range_bins_path)
            save_pickle(range_error_model_input_format, range_model_input_format_path)
            save_pickle(range_model_params, range_model_params_path)
            save_pickle(detected_range_uncertainty_model_params, detected_range_model_params_path)
 
        else:
            print("Loading sensor uncertainty models from SUM/ cache...")
 
            # === Load Uncertainty Model Function State Data ===
            pixel_uncertainty_model_state_data = tuple(load_pickle(pixel_model_state_path))
            range_bins = load_numpy(range_bins_path)
            range_error_model_input_format = load_pickle(range_model_input_format_path)
            range_model_params = load_pickle(range_model_params_path)
            detected_range_uncertainty_model_params = load_pickle(detected_range_model_params_path)
 
            # === Rebuild Uncertainty Model Functions ===
            pixel_uncertainty_model = generate_pixel_uncertainty_model(bins=20, pregen_state_data = pixel_uncertainty_model_state_data)
            range_uncertainty_model = generate_range_uncertainty_model(range_bins = range_bins, error_trim=2500, pregen_state_data = range_error_model_input_format, pregen_model_params = range_model_params)
    
    ###############################################
    ######## Initialise Encounter Array  #########
    ###############################################
 
    encounters = {}
 
    ###############################################
    ######## Initialise Cameras, Ownship and Encounter #########
    ###############################################
 
    # Define Camera Parameters
    cam_fovs = [
        (61.76, 48.26), # Front-Left
        (68.4,  53.96), # Side-Left
        (76.4,  61.04), # Rear-Left
        (61.76, 48.26), # Front-Right
        (68.4,  53.96), # Side-Right
        (76.4,  61.04) # Rear-Right
    ]
 
    cam_res_new = (9344, 7000)
 
    cam_positions = [(0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0)]
 
    cam_orientations = [(0.0, 0.0, 329.62),
                            (0.0, 0.0, 265.04),
                            (0.0, 0.0, 193.14),
                            (0.0, 0.0, 30.38),
                            (0.0, 0.0, 94.96),
                            (0.0, 0.0, 166.86)]
    
    # Create Dictionary of Cameras
    cameras = {
                i:
                    {
                        'res': cam_res_new,
                        'fov': cam_fovs[i],
                        'pos': cam_positions[i],
                        'orientation': cam_orientations[i],
                        'cam': Camera(cam_res_new, cam_fovs[i])
                    }
                for i in range(6)
            }
 
    # Initialise Ownship
    ownship_id = "Ownship_1"
    ownship = Ownship(ownship_id)
 
    # Initialise Camera Uncertainty Models with Camera Parameters and add to Ownship
    for camera in cameras.values():
        if SUM_ON:
            camera['cam'].add_uncertainty_models(pixel_uncertainty_model, range_uncertainty_model, detected_range_uncertainty_model_params, range_error_model_input_format)
        ownship.add_camera(camera['cam'], camera['pos'], camera['orientation'])

    # Mitigated (with Daidalus)
    main_mitigated = Main()
    main_mitigated.ownship = ownship
    main_mitigated.encounters = encounters
    main_mitigated.use_daidalus = True
    main_mitigated.run(ownship_file, intruder_file, [20])
    mitigated_stats = main_mitigated.statistics
    mitigated_summary = mitigated_stats.getTotalStats()
    mitigated_summary['files'] = [ownship_file, intruder_file]
    with open("results/mitigated_summary.json", "w") as f:
        json.dump(mitigated_summary, f, indent=2)

    # Compare
    """ from airspaceStatistics import AirspaceStatistics
    comparison = AirspaceStatistics.compare_runs(baseline_stats, mitigated_stats)
    comparison['files'] = [ownship_file, intruder_file]

    os.makedirs('results', exist_ok=True)
    with open('results/comparison_summary.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\n===== BASELINE STATS =====")
    print(json.dumps(baseline_summary, indent=2))
    print("\n===== MITIGATED STATS =====")
    print(json.dumps(mitigated_summary, indent=2))
    print("\n===== COMPARISON =====")
    print(json.dumps(comparison, indent=2)) """
