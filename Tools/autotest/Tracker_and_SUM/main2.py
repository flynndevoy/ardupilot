from Tracker_and_SUM.daidalusClient import DaidalusClient
import scipy
import numpy as np
import time
import navpy
from Tracker_and_SUM.airspaceStatistics import AirspaceStatistics
from Tracker_and_SUM.sixDOFSim import SixDOFSim
import mat73
import json
import os
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import threading
import hashlib
import orjson
import copy

# Now you can import
from Tracker_and_SUM.tracker import Tracker
from Tracker_and_SUM.synthetic_encounter_classes import Encounter
from Tracker_and_SUM.async_daidalus_client import Guidance_Thread

# State hashing function for keeping track of state updates in each thread

_ORJSON_OPTS = (
    orjson.OPT_SORT_KEYS       # stable ordering
    | orjson.OPT_SERIALIZE_NUMPY  # handles numpy arrays/scalars
    | orjson.OPT_NON_STR_KEYS  # if any dict keys aren’t strings
)

def state_id_fingerprint(obj, digest_size: int = 16) -> str:
        b = orjson.dumps(obj, option=_ORJSON_OPTS)
        return hashlib.blake2b(b, digest_size=digest_size).hexdigest()

class Main():
    
    def __init__(self) -> None:
        self.TARGET_LOOP_UPDATE_FREQ = 1/0.1 #100000000 #hz
        self.target_loop_wait_time = 1.0/self.TARGET_LOOP_UPDATE_FREQ
        
        self.dt = 0.1 # actual timestep in matlab file

        self.intruder_track_id = 1 # start at 1

        self.statistics = AirspaceStatistics()

        self.use_daidalus = True

        self.async_guidance = True

        if self.async_guidance:
            # lock for shared datastore writes
            self._lock = threading.Lock()

    def save_and_plot_encounter(self, encounter_id, ownship_north_m, ownship_east_m, ownship_lla, intruder_lla_gt, intruder_lla_raw, intruder_lla_smoothed, intruder_lla_list_raw_dict, intruder_lla_list_smoothed_dict, track_ids, guidance_headings, detection_confidences):
        print("Creating Plots...")
        
        debugging = False
        presentation_plot = True

        if debugging:
            # === 2D North vs East Plot ===
            plt.figure(figsize=(8, 6))

            # Full track lines
            plt.plot(ownship_lla[:, 1], ownship_lla[:, 0], color="blue", label='Mitigated Ownship Track')
            plt.plot(ownship_east_m, ownship_north_m, color="blue", linestyle='--', label='Unmitigated Ownship Track')
            plt.plot(intruder_lla_gt[:, 1], intruder_lla_gt[:, 0], color='gray', label='Intruder Track Ground Truth')
            #plt.plot(intruder_lla_raw[:, 1], intruder_lla_raw[:, 0], color='gray', label='Intruder Track Raw', alpha=0.3)
            # plt.plot(intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 0], label='Intruder Track Smoothed')
            
            cmap = plt.get_cmap('tab10')  # Or use 'tab20', 'hsv', etc.

            for i, (track_id, intruder_lla_raw_trk) in enumerate(intruder_lla_list_raw_dict.items()):       
                
                north = np.array(intruder_lla_raw_trk)[:, 0]
                east = np.array(intruder_lla_raw_trk)[:, 1]
                vel_north = np.array(intruder_lla_raw_trk)[:, 3]
                vel_east = np.array(intruder_lla_raw_trk)[:, 4]

                plt.plot(
                    east, north,
                    color=cmap(i % cmap.N),
                    label=f'Raw Track ID: {track_id}',
                    alpha=0.3
                )

            for i, (track_id, intruder_lla_smoothed_trk) in enumerate(intruder_lla_list_smoothed_dict.items()):       
                
                north = np.array(intruder_lla_smoothed_trk)[:, 0]
                east = np.array(intruder_lla_smoothed_trk)[:, 1]
                vel_north = np.array(intruder_lla_smoothed_trk)[:, 3]
                vel_east = np.array(intruder_lla_smoothed_trk)[:, 4]

                plt.plot(
                    east, north,
                    color=cmap(i % cmap.N),
                    label=f'Smoothed Track ID: {track_id}',
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

            # === Draw lines between ownship and intruder every N points ===
            N = 50  # You can adjust this to any interval you like
            min_len = min(len(ownship_lla), len(intruder_lla_gt))

            for i in range(0, min_len, N):
                plt.plot(
                    [ownship_lla[i, 1], intruder_lla_gt[i, 1]],  # East
                    [ownship_lla[i, 0], intruder_lla_gt[i, 0]],  # North
                    color='gray',
                    alpha=0.3,
                    linewidth=1,
                    linestyle = '--'
                )

            # Plot Guidance Heading Arrows

            # Convert headings to radians (convert from clockwise-from-North to counterclockwise-from-X)
            headings_rad = np.radians(guidance_headings)  # Rotate to match quiver's coord system

            # Compute dx, dy unit vectors (nan-safe)
            dx = np.sin(headings_rad)
            dy = np.cos(headings_rad)

            # Mask out NaNs in heading (i.e., don't plot arrows there)
            valid = ~np.isnan(guidance_headings)

            # Plot quiver
            plt.quiver(
                ownship_lla[valid, 1],
                ownship_lla[valid, 0],
                dx[valid],
                dy[valid],
                angles='xy',
                scale_units='xy',
                scale=1/200,
                width=0.0025,
                alpha=0.7,
                color='red',
                label='Heading Guidance'
            )

            # Start points
            plt.scatter(ownship_lla[0, 1], ownship_lla[0, 0], color='green', marker='o', s= 80, label='Ownship Start')
            #plt.scatter(intruder_lla_smoothed[0, 1], intruder_lla_smoothed[0, 0], color='green', marker='o', s= 80, label='Smoothed Intruder Start')
            plt.scatter(intruder_lla_gt[0, 1], intruder_lla_gt[0, 0], color='green', marker='x', s= 80, label='Intruder Start')

            # End points
            plt.scatter(ownship_lla[-1, 1], ownship_lla[-1, 0], color='red', marker='o', s= 80,  label='Ownship End')
            #plt.scatter(intruder_lla_smoothed[-1, 1], intruder_lla_smoothed[-1, 0], color='red', marker='o', s= 80,  label='Smoothed Intruder End')
            plt.scatter(intruder_lla_gt[-1, 1], intruder_lla_gt[-1, 0], color='red', marker='x', s= 80,  label='Intruder End')

            plt.xlabel('East (m)')
            plt.ylabel('North (m)')
            filename = os.path.basename(__file__) if '__file__' in globals() else 'interactive_session'
            plt.title(f'Encounter {encounter_id}: North vs East ({filename}), ({self.coord_frame})')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')

            plt.show()

                    # track_ids_unique = np.unique(track_ids)  # Get unique track IDs
            # cmap = plt.get_cmap('tab10')  # Or use 'tab20', 'hsv', etc.

            # for i, track_id in enumerate(track_ids_unique):
            #     mask = track_ids == track_id
                
            #     north = intruder_lla_smoothed[mask, 0]
            #     east = intruder_lla_smoothed[mask, 1]
            #     vel_north = intruder_lla_smoothed[mask, 3]
            #     vel_east = intruder_lla_smoothed[mask, 4]

            #     plt.plot(
            #         east, north,
            #         color=cmap(i % cmap.N),
            #         label=f'Smoothed Track ID: {track_id}'
            #     )

            #     # Add velocity vectors using quiver
            #     plt.quiver(
            #         east, north,       # Positions
            #         vel_east, vel_north,   # Components of velocity (U=East, V=North)
            #         color=cmap(i % cmap.N),
            #         angles='xy',
            #         scale_units='xy',
            #         scale=1,
            #         width=0.0025,
            #         alpha=0.7
            #     )

        if presentation_plot:
            fused = True
            sum_on = True
            if sum_on:
                if fused:
                    # === 2D North vs East Plot ===
                    fig = plt.figure(figsize=(16, 6))
                    cmap = plt.get_cmap('tab10')  # Or use 'tab20', 'hsv', etc.

                    ax1 = fig.add_subplot(1, 2, 1)

                    # Full track lines
                    ax1.plot(ownship_lla[:, 1], ownship_lla[:, 0], color="blue", label='Mitigated Ownship Track')
                    ax1.plot(ownship_east_m, ownship_north_m, color="blue", alpha=0.4, linestyle='--', label='Unmitigated Ownship Track')
                    ax1.plot(intruder_lla_gt[:, 1], intruder_lla_gt[:, 0], color='gray', label='Intruder Track Ground Truth')
                    #ax1.plot(intruder_lla_raw[:, 1], intruder_lla_raw[:, 0], color='gray', label='Intruder Track Raw', alpha=0.3)
                    # ax1.plot(intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 0], label='Intruder Track Smoothed')

                    for i, (track_id, intruder_lla_raw_trk) in enumerate(intruder_lla_list_raw_dict.items()):       
                        
                        north = np.array(intruder_lla_raw_trk)[:, 0]
                        east = np.array(intruder_lla_raw_trk)[:, 1]
                        vel_north = np.array(intruder_lla_raw_trk)[:, 3]
                        vel_east = np.array(intruder_lla_raw_trk)[:, 4]

                        ax1.plot(
                            east, north,
                            color=cmap(i % cmap.N),
                            label=f'Raw Track: {track_id}',
                            alpha=0.3
                        )

                    for i, (track_id, intruder_lla_smoothed_trk) in enumerate(intruder_lla_list_smoothed_dict.items()):       
                        
                        north = np.array(intruder_lla_smoothed_trk)[:, 0]
                        east = np.array(intruder_lla_smoothed_trk)[:, 1]
                        vel_north = np.array(intruder_lla_smoothed_trk)[:, 3]
                        vel_east = np.array(intruder_lla_smoothed_trk)[:, 4]

                        if i == 0:
                            ax1.plot(
                                east, north,
                                color=cmap(0),
                                label=f'Fused Track'
                            )
                        else:
                            ax1.plot(
                                east, north,
                                color=cmap(0)
                            )

                        # Add velocity vectors at fixed separation interval using quiver
                        N = 10  # Arrow sample interval

                        # Subsample every N points
                        east_sub = east[::N]
                        north_sub = north[::N]
                        vel_east_sub = vel_east[::N]
                        vel_north_sub = vel_north[::N]

                        # Add velocity vectors using quiver
                        ax1.quiver(
                            east_sub, north_sub,       # Positions
                            vel_east_sub, vel_north_sub,   # Components of velocity (U=East, V=North)
                            color=cmap(i % cmap.N),
                            angles='xy',
                            scale_units='xy',
                            scale=1/3,
                            width=0.0015,
                            alpha=1
                        )

                    # === Draw lines between ownship and intruder every N points ===
                    N = 50  # Line sample interval
                    min_len = min(len(ownship_lla), len(intruder_lla_gt))

                    for i in range(0, min_len, N):
                        ax1.plot(
                            [ownship_lla[i, 1], intruder_lla_gt[i, 1]],  # East
                            [ownship_lla[i, 0], intruder_lla_gt[i, 0]],  # North
                            color='gray',
                            alpha=0.3,
                            linewidth=1,
                            linestyle = '--'
                        )

                    # Plot Guidance Heading Arrows

                    import matplotlib.lines as mlines

                    # Create a legend handle that looks like an arrow
                    arrow_handle = mlines.Line2D(
                        [], [], 
                        color='red',
                        marker=r'$\rightarrow$', 
                        linestyle='None',
                        markersize=12,
                        markeredgewidth=0.05,  # Thinner line weight
                        label='Heading Guidance'
                    )

                    N = 4  # Arrow display interval

                    # Mask out invalid entries (e.g., NaN or false)
                    valid = (~np.isnan(guidance_headings)) & (guidance_headings != False)

                    # Convert headings to radians
                    headings_rad_valid = np.radians(guidance_headings[valid])

                    # Compute unit vectors
                    dx = np.sin(headings_rad_valid)
                    dy = np.cos(headings_rad_valid)

                    # Apply mask
                    lat = ownship_lla[valid, 0]
                    lon = ownship_lla[valid, 1]

                    # Subsample every Nth valid entry
                    lat_sub = lat[::N]
                    lon_sub = lon[::N]
                    dx_sub = dx[::N]
                    dy_sub = dy[::N]

                    # Plot quiver
                    ax1.quiver(
                        lon_sub,
                        lat_sub,
                        dx_sub,
                        dy_sub,
                        angles='xy',
                        scale_units='xy',
                        scale=1/200,
                        width=0.0015,
                        alpha=0.7,
                        color='red',
                    )

                    # Lat/lon for saturated points
                    guidance_saturated = guidance_headings == False
                    sat_lat = ownship_lla[guidance_saturated, 0]
                    sat_lon = ownship_lla[guidance_saturated, 1]

                    # Plot large black triangles at saturated locations
                    ax1.scatter(
                        sat_lon,
                        sat_lat,
                        marker='^',        # triangle
                        s=120,             # size in points^2; bump up if you want larger
                        c='black',
                        edgecolors='white',
                        linewidths=0.6,
                        alpha=0.95,
                        zorder=5,
                        label='Guidance saturated'
                    )

                    # === Compute 2D distances between time-aligned points ===
                    min_len = min(len(ownship_lla), len(intruder_lla_gt))
                    own_xy = ownship_lla[:min_len, :2]  # [North, East]
                    intr_xy = intruder_lla_gt[:min_len, :2]

                    # Compute squared distances for each time-aligned pair
                    deltas = own_xy - intr_xy
                    dists_sq = np.sum(deltas**2, axis=1)
                    i_min = np.argmin(dists_sq)
                    min_dist = np.sqrt(dists_sq[i_min])

                    # Extract closest time-aligned pair
                    own_closest = own_xy[i_min]
                    intr_closest = intr_xy[i_min]

                    # === Plot HMD line ===
                    ax1.plot(
                        [own_closest[1], intr_closest[1]],  # East
                        [own_closest[0], intr_closest[0]],  # North
                        color='orange',
                        linestyle='--',
                        linewidth=1.5,
                        label=f'Horizontal Miss Distance = {int(min_dist)}m'
                    )

                    # Start points
                    ax1.scatter(ownship_lla[0, 1], ownship_lla[0, 0], color='green', marker='o', s= 80, label='Ownship Start')
                    #ax1.scatter(intruder_lla_smoothed[0, 1], intruder_lla_smoothed[0, 0], color='green', marker='o', s= 80, label='Smoothed Intruder Start')
                    ax1.scatter(intruder_lla_gt[0, 1], intruder_lla_gt[0, 0], color='green', marker='x', s= 80, label='Intruder Start')

                    # End points
                    ax1.scatter(ownship_lla[-1, 1], ownship_lla[-1, 0], color='red', marker='o', s= 80,  label='Ownship End')
                    #ax1.scatter(intruder_lla_smoothed[-1, 1], intruder_lla_smoothed[-1, 0], color='red', marker='o', s= 80,  label='Smoothed Intruder End')
                    ax1.scatter(intruder_lla_gt[-1, 1], intruder_lla_gt[-1, 0], color='red', marker='x', s= 80,  label='Intruder End')

                    ax1.set_xlabel('East (m)')
                    ax1.set_ylabel('North (m)')
                    filename = os.path.basename(__file__) if '__file__' in globals() else 'interactive_session'
                    #ax1.title(f'Encounter {encounter_id}: North vs East ({filename}), ({self.coord_frame})')
                    ax1.set_title("Multi-Camera Encounter (Fused Tracks)")
                    # Add to legend manually
                    ax1.legend(handles=[arrow_handle, *plt.gca().get_legend_handles_labels()[0]])
                    # ax1.legend()
                    ax1.grid(True)
                    ax1.axis('equal')

                    
                    # === Right subplot: 3D Confidence Plot ===
                    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

                    # Track whether labels have been added
                    labels_added = {'above': False, 'below': False, 'coasted': False}

                    for i, (conf_list, horizon_list) in enumerate(detection_confidences):
                        east = intruder_lla_gt[i, 1]
                        north = intruder_lla_gt[i, 0]

                        if conf_list is None or horizon_list is None:
                            # Plot orange dot for coasted
                            label = 'Coasted' if not labels_added['coasted'] else None
                            ax2.scatter(east, north, 1.0, color='orange', s=5, label=label)
                            labels_added['coasted'] = True
                            continue

                        # Ensure matched lengths
                        min_len = min(len(conf_list), len(horizon_list))
                        conf_list = conf_list[:min_len]
                        horizon_list = horizon_list[:min_len]

                        for conf, horizon in zip(conf_list, horizon_list):
                            if horizon == 1:
                                color = 'blue'
                                label = 'Above Horizon' if not labels_added['above'] else None
                                labels_added['above'] = True
                            else:
                                color = 'green'
                                label = 'Below Horizon' if not labels_added['below'] else None
                                labels_added['below'] = True

                            ax2.scatter(east, north, conf, color=color, s=5, label=label)

                    ax2.set_xlabel('East (m)')
                    ax2.set_ylabel('North (m)')
                    ax2.set_zlabel('Detection Confidence')
                    ax2.set_title('3D Detection Confidence (Above/Below Horizon/Coasted)')
                    ax2.view_init(elev=25, azim=175)
                    ax2.set_zlim([0, 1.0])

                    plt.show()
                else:
                    # === 2D North vs East Plot ===
                    plt.figure(figsize=(8, 6))

                    # Full track lines
                    plt.plot(ownship_lla[:, 1], ownship_lla[:, 0], color="blue", label='Mitigated Ownship Track')
                    plt.plot(ownship_east_m, ownship_north_m, color="blue", alpha=0.4, linestyle='--', label='Unmitigated Ownship Track')
                    plt.plot(intruder_lla_gt[:, 1], intruder_lla_gt[:, 0], color='gray', label='Intruder Track Ground Truth')
                    #plt.plot(intruder_lla_raw[:, 1], intruder_lla_raw[:, 0], color='gray', label='Intruder Track Raw', alpha=0.3)
                    # plt.plot(intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 0], label='Intruder Track Smoothed')
                    
                    cmap = plt.get_cmap('tab10')  # Or use 'tab20', 'hsv', etc.

                    for i, (track_id, intruder_lla_raw_trk) in enumerate(intruder_lla_list_raw_dict.items()):       
                        
                        north = np.array(intruder_lla_raw_trk)[:, 0]
                        east = np.array(intruder_lla_raw_trk)[:, 1]
                        vel_north = np.array(intruder_lla_raw_trk)[:, 3]
                        vel_east = np.array(intruder_lla_raw_trk)[:, 4]

                        plt.plot(
                            east, north,
                            color=cmap(i % cmap.N),
                            label=f'Raw Track: {track_id}',
                            alpha=0.3
                        )

                    for i, (track_id, intruder_lla_smoothed_trk) in enumerate(intruder_lla_list_smoothed_dict.items()):       
                        
                        north = np.array(intruder_lla_smoothed_trk)[:, 0]
                        east = np.array(intruder_lla_smoothed_trk)[:, 1]
                        vel_north = np.array(intruder_lla_smoothed_trk)[:, 3]
                        vel_east = np.array(intruder_lla_smoothed_trk)[:, 4]

                        plt.plot(
                            east, north,
                            color=cmap(i % cmap.N),
                            label=f'Smoothed Track: {track_id}',
                        )

                        # Add velocity vectors at fixed separation interval using quiver
                        N = 10  # Arrow sample interval

                        # Subsample every N points
                        east_sub = east[::N]
                        north_sub = north[::N]
                        vel_east_sub = vel_east[::N]
                        vel_north_sub = vel_north[::N]

                        # Add velocity vectors using quiver
                        plt.quiver(
                            east_sub, north_sub,       # Positions
                            vel_east_sub, vel_north_sub,   # Components of velocity (U=East, V=North)
                            color=cmap(i % cmap.N),
                            angles='xy',
                            scale_units='xy',
                            scale=1/3,
                            width=0.0015,
                            alpha=1
                        )

                    # === Draw lines between ownship and intruder every N points ===
                    N = 50  # Line sample interval
                    min_len = min(len(ownship_lla), len(intruder_lla_gt))

                    for i in range(0, min_len, N):
                        plt.plot(
                            [ownship_lla[i, 1], intruder_lla_gt[i, 1]],  # East
                            [ownship_lla[i, 0], intruder_lla_gt[i, 0]],  # North
                            color='gray',
                            alpha=0.3,
                            linewidth=1,
                            linestyle = '--'
                        )

                    # Plot Guidance Heading Arrows

                    import matplotlib.lines as mlines

                    # Create a legend handle that looks like an arrow
                    arrow_handle = mlines.Line2D(
                        [], [], 
                        color='red',
                        marker=r'$\rightarrow$', 
                        linestyle='None',
                        markersize=12,
                        markeredgewidth=0.05,  # Thinner line weight
                        label='Heading Guidance'
                    )

                    N = 4  # Arrow display interval

                    # Convert headings to radians
                    headings_rad = np.radians(guidance_headings)

                    # Compute unit vectors
                    dx = np.sin(headings_rad)
                    dy = np.cos(headings_rad)

                    # Mask out invalid entries (e.g., NaN)
                    valid = ~np.isnan(guidance_headings)

                    # Apply mask
                    lat = ownship_lla[valid, 0]
                    lon = ownship_lla[valid, 1]
                    dx_valid = dx[valid]
                    dy_valid = dy[valid]

                    # Subsample every Nth valid entry
                    lat_sub = lat[::N]
                    lon_sub = lon[::N]
                    dx_sub = dx_valid[::N]
                    dy_sub = dy_valid[::N]

                    # Plot quiver
                    plt.quiver(
                        lon_sub,
                        lat_sub,
                        dx_sub,
                        dy_sub,
                        angles='xy',
                        scale_units='xy',
                        scale=1/200,
                        width=0.0015,
                        alpha=0.7,
                        color='red',
                    )

                    # === Compute 2D distances between time-aligned points ===
                    min_len = min(len(ownship_lla), len(intruder_lla_gt))
                    own_xy = ownship_lla[:min_len, :2]  # [North, East]
                    intr_xy = intruder_lla_gt[:min_len, :2]

                    # Compute squared distances between all point pairs
                    min_dist = float('inf')
                    i_min, j_min = 0, 0

                    for i in range(min_len):
                        for j in range(min_len):
                            dy = own_xy[i, 0] - intr_xy[j, 0]  # North diff
                            dx = own_xy[i, 1] - intr_xy[j, 1]  # East diff
                            dist_sq = dx**2 + dy**2
                            if dist_sq < min_dist:
                                min_dist = dist_sq
                                i_min, j_min = i, j

                    # Extract the closest points
                    own_closest = own_xy[i_min]
                    intr_closest = intr_xy[j_min]

                    # === Plot HMD line ===
                    plt.plot(
                        [own_closest[1], intr_closest[1]],  # East
                        [own_closest[0], intr_closest[0]],  # North
                        color='orange',
                        linestyle='--',
                        linewidth=1.5,
                        label=f'Horizontal Miss Distance = {int(np.linalg.norm(np.array(own_closest) - np.array(intr_closest)))}m'
                    )

                    # Start points
                    plt.scatter(ownship_lla[0, 1], ownship_lla[0, 0], color='green', marker='o', s= 80, label='Ownship Start')
                    #plt.scatter(intruder_lla_smoothed[0, 1], intruder_lla_smoothed[0, 0], color='green', marker='o', s= 80, label='Smoothed Intruder Start')
                    plt.scatter(intruder_lla_gt[0, 1], intruder_lla_gt[0, 0], color='green', marker='x', s= 80, label='Intruder Start')

                    # End points
                    plt.scatter(ownship_lla[-1, 1], ownship_lla[-1, 0], color='red', marker='o', s= 80,  label='Ownship End')
                    #plt.scatter(intruder_lla_smoothed[-1, 1], intruder_lla_smoothed[-1, 0], color='red', marker='o', s= 80,  label='Smoothed Intruder End')
                    plt.scatter(intruder_lla_gt[-1, 1], intruder_lla_gt[-1, 0], color='red', marker='x', s= 80,  label='Intruder End')

                    plt.xlabel('East (m)')
                    plt.ylabel('North (m)')
                    filename = os.path.basename(__file__) if '__file__' in globals() else 'interactive_session'
                    #plt.title(f'Encounter {encounter_id}: North vs East ({filename}), ({self.coord_frame})')
                    plt.title("Multi-Camera Encounter (Unfused Tracks)")
                    # Add to legend manually
                    plt.legend(handles=[arrow_handle, *plt.gca().get_legend_handles_labels()[0]])
                    # plt.legend()
                    plt.grid(True)
                    plt.axis('equal')

                    plt.show()
            else:
                # === 2D North vs East Plot ===
                plt.figure(figsize=(8, 6))

                # Full track lines
                plt.plot(ownship_lla[:, 1], ownship_lla[:, 0], color="blue", label='Mitigated Ownship Track')
                plt.plot(ownship_east_m, ownship_north_m, color="blue", alpha=0.4, linestyle='--', label='Unmitigated Ownship Track')
                plt.plot(intruder_lla_gt[:, 1], intruder_lla_gt[:, 0], color='gray', label='Intruder Track Ground Truth')
                #plt.plot(intruder_lla_raw[:, 1], intruder_lla_raw[:, 0], color='gray', label='Intruder Track Raw', alpha=0.3)
                # plt.plot(intruder_lla_smoothed[:, 1], intruder_lla_smoothed[:, 0], label='Intruder Track Smoothed')

                for i, (track_id, intruder_lla_smoothed_trk) in enumerate(intruder_lla_list_smoothed_dict.items()):       
                    
                    north = np.array(intruder_lla_smoothed_trk)[:, 0]
                    east = np.array(intruder_lla_smoothed_trk)[:, 1]
                    vel_north = np.array(intruder_lla_smoothed_trk)[:, 3]
                    vel_east = np.array(intruder_lla_smoothed_trk)[:, 4]

                    # Add velocity vectors at fixed separation interval using quiver
                    N = 10  # Arrow sample interval

                    # Subsample every N points
                    east_sub = east[::N]
                    north_sub = north[::N]
                    vel_east_sub = vel_east[::N]
                    vel_north_sub = vel_north[::N]

                    # Add velocity vectors using quiver
                    plt.quiver(
                        east_sub, north_sub,       # Positions
                        vel_east_sub, vel_north_sub,   # Components of velocity (U=East, V=North)
                        color='blue',
                        angles='xy',
                        scale_units='xy',
                        scale=1/3,
                        width=0.0015,
                        alpha=1
                    )

                # === Draw lines between ownship and intruder every N points ===
                N = 50  # Line sample interval
                min_len = min(len(ownship_lla), len(intruder_lla_gt))

                for i in range(0, min_len, N):
                    plt.plot(
                        [ownship_lla[i, 1], intruder_lla_gt[i, 1]],  # East
                        [ownship_lla[i, 0], intruder_lla_gt[i, 0]],  # North
                        color='gray',
                        alpha=0.3,
                        linewidth=1,
                        linestyle = '--'
                    )

                # Plot Guidance Heading Arrows

                import matplotlib.lines as mlines

                # Create a legend handle that looks like an arrow
                arrow_handle = mlines.Line2D(
                    [], [], 
                    color='red',
                    marker=r'$\rightarrow$', 
                    linestyle='None',
                    markersize=12,
                    markeredgewidth=0.05,  # Thinner line weight
                    label='Heading Guidance'
                )

                N = 4  # Arrow display interval

                # Convert headings to radians
                headings_rad = np.radians(guidance_headings)

                # Compute unit vectors
                dx = np.sin(headings_rad)
                dy = np.cos(headings_rad)

                # Mask out invalid entries (e.g., NaN)
                valid = ~np.isnan(guidance_headings)

                # Apply mask
                lat = ownship_lla[valid, 0]
                lon = ownship_lla[valid, 1]
                dx_valid = dx[valid]
                dy_valid = dy[valid]

                # Subsample every Nth valid entry
                lat_sub = lat[::N]
                lon_sub = lon[::N]
                dx_sub = dx_valid[::N]
                dy_sub = dy_valid[::N]

                # Plot quiver
                plt.quiver(
                    lon_sub,
                    lat_sub,
                    dx_sub,
                    dy_sub,
                    angles='xy',
                    scale_units='xy',
                    scale=1/200,
                    width=0.0015,
                    alpha=0.7,
                    color='red',
                )

                # === Compute 2D distances between time-aligned points ===
                min_len = min(len(ownship_lla), len(intruder_lla_gt))
                own_xy = ownship_lla[:min_len, :2]  # [North, East]
                intr_xy = intruder_lla_gt[:min_len, :2]

                # Compute squared distances for each time-aligned pair
                deltas = own_xy - intr_xy
                dists_sq = np.sum(deltas**2, axis=1)
                i_min = np.argmin(dists_sq)
                min_dist = np.sqrt(dists_sq[i_min])

                # Extract closest time-aligned pair
                own_closest = own_xy[i_min]
                intr_closest = intr_xy[i_min]

                # === Plot HMD line ===
                plt.plot(
                    [own_closest[1], intr_closest[1]],  # East
                    [own_closest[0], intr_closest[0]],  # North
                    color='orange',
                    linestyle='--',
                    linewidth=1.5,
                    label=f'Horizontal Miss Distance = {int(min_dist)}m'
                )

                # Start points
                plt.scatter(ownship_lla[0, 1], ownship_lla[0, 0], color='green', marker='o', s= 80, label='Ownship Start')
                #plt.scatter(intruder_lla_smoothed[0, 1], intruder_lla_smoothed[0, 0], color='green', marker='o', s= 80, label='Smoothed Intruder Start')
                plt.scatter(intruder_lla_gt[0, 1], intruder_lla_gt[0, 0], color='green', marker='x', s= 80, label='Intruder Start')

                # End points
                plt.scatter(ownship_lla[-1, 1], ownship_lla[-1, 0], color='red', marker='o', s= 80,  label='Ownship End')
                #plt.scatter(intruder_lla_smoothed[-1, 1], intruder_lla_smoothed[-1, 0], color='red', marker='o', s= 80,  label='Smoothed Intruder End')
                plt.scatter(intruder_lla_gt[-1, 1], intruder_lla_gt[-1, 0], color='red', marker='x', s= 80,  label='Intruder End')

                plt.xlabel('East (m)')
                plt.ylabel('North (m)')
                filename = os.path.basename(__file__) if '__file__' in globals() else 'interactive_session'
                #plt.title(f'Encounter {encounter_id}: North vs East ({filename}), ({self.coord_frame})')
                plt.title("Ideal Encounter (No Sensor Uncertainty))")
                # Add to legend manually
                plt.legend(handles=[arrow_handle, *plt.gca().get_legend_handles_labels()[0]])
                # plt.legend()
                plt.grid(True)
                plt.axis('equal')

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

                    self.datastore = {
                        # Stores the last updated tracker output state for passing into daidalus
                        'track': {
                            'processed_ownship_state': None,
                            'track_items': None,
                            'state_id': None,
                        },
                        # Daidalus free flag
                        'daidalus_free': True,
                        'daidalus_last_call_state_id': None,
                        'daidalus_input': {},
                        # Stores last daidalus guidance to send to sim
                        'daidalus_out': {
                            'horizontal_guidance': [],
                            'altitude_guidance': [],
                            'horizontal_speed_guidance': [],
                            'vertical_speed_guidance': [],
                            'intruders': [],
                            'suggested_hdir_deg': np.nan,
                            'suggested_hspd_knot': np.nan,
                            'suggested_vspd_fpm': np.nan,
                            'suggested_alt_ft': np.nan,
                            'timestamp': None
                        }
                    }

                    self.results[encounter_idx] = []
                    tracker = Tracker(ownship_id = f"OS{encounter_idx}")
                    enc_this = Encounter(tracker.ownship)
 
                    self.start_simulating_ownship = False
                    self.waited_for_pilot_reponse = False
                    self.ownship_sim = None
                    start_sim_idx = None

                    coord_frame_configs = tracker.load_camera_config_data()['daidalus_input_coord_frame']
                    cartesian, centered, aligned = coord_frame_configs['cartesian'], coord_frame_configs['centered'], coord_frame_configs['aligned']

                    if cartesian:
                        if aligned:
                            self.coord_frame = "NEU_aligned"
                        elif centered:
                            self.coord_frame = "NEU_centered"
                        else:
                            self.coord_frame = "NEU"

                        if aligned and not centered:
                            print("Must center if aligning!")
                            error_out_here
                    else:
                        self.coord_frame = "LLA"

                    ownship_lla_list = []
                    intruder_lla_list_gt = []
                    intruder_lla_list_raw = []
                    intruder_lla_list_raw_dict = {}
                    intruder_lla_list_smoothed = []
                    intruder_lla_list_smoothed_dict = {}
                    track_ids = []
                    guidance = []
                    input_coords = []
                    detection_confidences = []
                    daidalus_outputs = []
                    daidalus_guidance = [np.nan, np.nan, np.nan, np.nan]

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

                    if self.async_guidance:
                        # Plain sync function that returns a dict
                        def send_fn(payload: dict) -> dict:
                            return self.client.send_data2(connection, payload)

                        ge = Guidance_Thread(send_fn=send_fn, tracker=tracker, datastore=self.datastore)
                        ge.start()

                    for row_idx in range(len(ownship_east_m)-1):
                        start = time.time()

                        ownship_heading_deg = float(np.degrees(ownship_hdg_rad[row_idx]))
                        ownship_heading_deg = (ownship_heading_deg + 360) % 360
                        # ownship_vx = ownship_speed_m_per_s[idx] * np.cos(ownship_hdg_rad[idx])
                        # ownship_vy = ownship_speed_m_per_s[idx] * np.sin(ownship_hdg_rad[idx]) 
                        # Generate initial ownship LLA Coords 
                        ownship_lla = navpy.ned2lla([float(ownship_north_m[row_idx]), float(ownship_east_m[row_idx]), float(ownship_up_m[row_idx])], self.ref_lla[0], self.ref_lla[1], self.ref_lla[2])
                        ownship_ver_vel = float(ownship_vz[row_idx])
                        ownship_hor_vel = float(ownship_speed_m_per_s[row_idx])
                        ownship_time = ownship_time_sec[row_idx]
                        ownship_phi = float(np.degrees(ownship_phi_rad[row_idx]))
                        ownship_theta = float(np.degrees(ownship_theta_rad[row_idx]))
                        ownship_psi = float(np.degrees(ownship_hdg_rad[row_idx]))
                        print(ownship_time)
                        if not home_lla_recorded:
                            home_lla = ownship_lla
                            home_lla_recorded = True

                        if self.start_simulating_ownship:
                            self.ownship_sim.simulate()
                            
                            # Calculate ownship NEU relative to home location LLA
                            ownship_state = {"lat": self.ownship_sim.lat,
                                            "lon": self.ownship_sim.long,
                                            "alt": self.ownship_sim.alt,
                                            "trk": self.ownship_sim.track,
                                            "ground_speed": self.ownship_sim.speed,
                                            "vertical_speed": self.ownship_sim.vert_speed,
                                            "time": ownship_time,
                                            "rl": 0,#ownship_phi, # SUM (Roll)
                                            "pt": 0,#ownship_theta, # SUM (Pitch)
                                            "yw": self.ownship_sim.track, # SUM (Yaw)
                                            "wind_dir": 0,
                                            "wind_spd": 0
                            }
                        else:
                            # Populate ownship lat lon with the original LLA Coords using ref_lla
                            ownship_state = {"lat": ownship_lla[0],
                                            "lon": ownship_lla[1],
                                            "alt": ownship_up_m[row_idx],
                                            "trk": ownship_heading_deg,
                                            "ground_speed": ownship_hor_vel,
                                            "vertical_speed": ownship_ver_vel,
                                            "time": ownship_time,
                                            "rl": 0,#ownship_phi, # SUM (Roll)
                                            "pt": 0,#ownship_theta, # SUM (Pitch)
                                            "yw": ownship_psi, # SUM (Yaw)
                            }
                            
                        # - Intruder State -
                        intruder_states = []
                        intruder_heading_deg = float(np.degrees(intruder_hdg_rad[row_idx]))
                        intruder_heading_deg = (intruder_heading_deg + 360) % 360

                        # Generate initial intruder LLA Coords based on the same ref_lla that ownship lla coords are based on
                        intruder_lla = navpy.ned2lla([float(intruder_north_m[row_idx]), float(intruder_east_m[row_idx]), float(-intruder_up_m[row_idx])], home_lla[0], home_lla[1], 0)
                        intruder_lla = (intruder_lla[0], intruder_lla[1], intruder_up_m[row_idx])
                        intruder_ver_vel = float(intruder_vz[row_idx])
                        intruder_hor_vel = float(intruder_speed_m_per_s[row_idx])

                        intruder_state = {"lat": intruder_lla[0],
                                        "lon": intruder_lla[1],
                                        "alt": intruder_lla[2], # add North, East, Up for SUM (as 'x', 'y', 'z')
                                        "trk": intruder_heading_deg,
                                        "north_m": intruder_north_m[row_idx],
                                        "east_m": intruder_east_m[row_idx],
                                        "ground_speed": intruder_hor_vel,
                                        "vertical_speed": intruder_ver_vel,
                                        "type": "DAA",
                                        "callsign": "INTRUDER",
                                        "id": str(encounter_idx),
                                        "time": ownship_time,
                        }

                        ##### Tracker (OFF SFF) ######

                        ###### Process Intruder and Ownship Input State #####
                        synthetic_ownship_mavlink_position, synthetic_ownship_mavlink_attitude = enc_this.get_ownship_mavlink_format(ownship_state)
                        synthetic_detector_output = enc_this.process_synthetic_data(row_idx, tracker.ownship, ownship_state, [intruder_state], home_lla)
                        
                        synthetic_adsb_intruders = [] # Generate mock ADS-B traffic data based on intruder format (can convert ADSB to this format)
                        
                        TRAF = {"lat": intruder_lla[0],
                                "lon": intruder_lla[1],
                                "alt": intruder_lla[2], # add North, East, Up for SUM (as 'x', 'y', 'z')
                                "trk": intruder_heading_deg,
                                "north_m": intruder_north_m[row_idx],
                                "east_m": intruder_east_m[row_idx],
                                "ground_speed": intruder_hor_vel,
                                "vertical_speed": intruder_ver_vel,
                                "type": "ADSB",
                                "callsign": "INTRUDER",
                                "id": str(encounter_idx),
                                "time": ownship_time,
                                "NIC": 8,
                                "NACp": 9
                                # "coord_frame": coord_frame,
                                # "centered": centered,
                                # "aligned": aligned
                            }
                        
                        #synthetic_adsb_intruders.append(TRAF)

                        ###############################################################################################################

                        ##### Tracker (ON SFF) ######

                        ##### Now generate ownship and intruder cartesian coordinates relative to the home_lla (aka ref_lla is not used past this point) #####

                        processed_ownship_state = tracker.process_ownship_state(synthetic_ownship_mavlink_position, synthetic_ownship_mavlink_attitude)

                        # Get the hash of the latest airspace picture (ownship + intruders) state
                        airspace_state_id = state_id_fingerprint([processed_ownship_state, synthetic_detector_output, synthetic_adsb_intruders])

                        # If matching hash with the previously saved hash, then nothing has changed and we can skip this update
                        if airspace_state_id == self.datastore['track']['state_id']:
                            pass
                        else:
                            ###################################
                            ###### TRACKER UPDATE LOGIC #######
                            ###################################

                            # Update tracker for updated airspace
                            new_track_points, all_current_intruder_tracks, deleted_track_ids = tracker.update_tracks_v3(processed_ownship_state, synthetic_detector_output, synthetic_adsb_intruders, intruder_state)#, daidalus_guidance)

                            # Extract kalman filtered ownship state
                            new_tracked_ownship_state = new_track_points.pop("Ownship", {})

                            # Update ownship state if new ownship object returned (aka. new ownship telemetry ingested)
                            if new_tracked_ownship_state:
                                tracked_ownship_state = new_tracked_ownship_state
                            else:
                                tracked_ownship_state = copy.deepcopy(self.datastore['track']['processed_ownship_state'])

                            #Update tracker datastore object with latest tracker output
                            with self._lock:
                                self.datastore['track'] = {
                                    'processed_ownship_state': tracked_ownship_state,
                                    'track_items': new_track_points,
                                    'state_id': airspace_state_id,
                                }

                            ###################################################
                            ###### DAIDALUS INPUT OBJECT CREATION LOGIC #######
                            ###################################################

                            # Check if the last daidalus output included guidance
                            last_daidalus_response_object = copy.deepcopy(self.datastore['daidalus_out'])
                            guidance_in_last_daidalus_object = (
                                last_daidalus_response_object['suggested_hdir_deg'] is not None and np.isfinite(last_daidalus_response_object['suggested_hdir_deg']) or
                                last_daidalus_response_object['suggested_hspd_knot'] is not None and np.isfinite(last_daidalus_response_object['suggested_hspd_knot']) or
                                last_daidalus_response_object['suggested_vspd_fpm'] is not None and np.isfinite(last_daidalus_response_object['suggested_vspd_fpm']) or
                                last_daidalus_response_object['suggested_alt_ft'] is not None and np.isfinite(last_daidalus_response_object['suggested_alt_ft'])
                            )

                            # Update daidalus input object if:
                            # (1) intruder tracks update or 
                            # (2) ownship track updates and there exist intruders or 
                            # (3) if tracks deleted and the last call included guidance (since this may eliminate guidance)
                            if (
                                new_track_points
                                or (new_tracked_ownship_state and all_current_intruder_tracks)
                                or (deleted_track_ids and guidance_in_last_daidalus_object)
                                ):
                                # If the last daidalus guidance output was not empty, reset iterations from last guidance to zero
                                # This is done inside this block since we want an iteration to be whenever the airspace changes and we make a new daidalus object
                                # rather than an iteration of the thread, which is arbitrary and much faster
                                for track_id_this in tracker.iso_decomp_guidance_hist:
                                    for daidalus_track_response in last_daidalus_response_object['intruders']:
                                        if str(daidalus_track_response['id']) == str(track_id_this):
                                            if daidalus_track_response['alert_level'] > 0:
                                                tracker.iso_decomp_guidance_hist[track_id_this]['iters_since_guidance'] = 0
                                            else:
                                                tracker.iso_decomp_guidance_hist[track_id_this]['iters_since_guidance'] += 1

                                daidalus_input = tracker.create_daidalus_input(tracked_ownship_state, all_current_intruder_tracks)
                                self.datastore['daidalus_input'] = daidalus_input

                            ####################################
                            ###### DAIDALUS UPDATE LOGIC #######
                            ####################################
                            daidalus_input = copy.deepcopy(self.datastore['daidalus_input'])

                            # Check if we have valid daidalus input data
                            if daidalus_input:
                                # Get the hash of the latest daidalus_input object
                                daidalus_input_state_id = state_id_fingerprint(daidalus_input)

                                # Only make daidalus call if daidalus input has changed from the last call
                                if daidalus_input_state_id == self.datastore['daidalus_last_call_state_id']:
                                    pass
                                else:                
                                    if self.async_guidance:
                                        #############################################################################
                                        ############## NEW ASYNC GUIDANCE AND EXECUTE IMPLEMENTATION ################
                                        #############################################################################
                                        # Only feed latest track state into daidalus if it's free
                                        if self.datastore['daidalus_free']:
                                            print("############################################################")
                                            print(daidalus_input)
                                            self.datastore['daidalus_last_call_state_id'] = daidalus_input_state_id
                                            print("NEW DIADLUS SUBMISSION")
                                            ge.submit(daidalus_input)
                                        else:
                                            print("DAIDALUS BUSY")
                                            pass
                                    else:
                                        #############################################################################
                                        ############## OLD SYNC GUIDANCE AND EXECUTE IMPLEMENTATION ################
                                        #############################################################################
                                        response = self.client.send_data2(connection, daidalus_input)
                                        self.datastore['daidalus_out'] = response
                        
                        # Pull data from current datastate instead of the output this iteration to mimic async loop architecture
                        response = copy.deepcopy(self.datastore['daidalus_out'])

                        ###########################
                        ###### DATA LOGGING #######
                        ###########################

                        # Update storage objects for plotting (since track state has changed)
                        ownship_lla_list.append([tracker.ownship.position()[0], tracker.ownship.position()[1], tracker.ownship.position()[2], ownship_state["ground_speed"], ownship_state["vertical_speed"], ownship_state["time"]])
                        intruder_lla_list_gt.append([intruder_north_m[row_idx], intruder_east_m[row_idx], intruder_up_m[row_idx], intruder_state["ground_speed"], intruder_state["vertical_speed"], intruder_state["time"]])

                        if new_track_points:
                            intruder_lla_list_raw.append([*list(new_track_points.values())[0]['abs_cartesian_x_y_z'], 0, 0, 0])
                            #intruder_lla_list_smoothed.append([*list(new_track_points.values())[0]['abs_cartesian_x_y_z_smoothed'], *list(new_track_points.values())[0]['velocity_x_y_z_smoothed']])
                            for track_id, detection_int in new_track_points.items():
                                intruder_lla_list_raw_dict.setdefault(track_id, []).append(
                                    [*list(detection_int['abs_cartesian_x_y_z']), *list(detection_int['velocity_x_y_z_smoothed'])]
                                    )
                                intruder_lla_list_smoothed_dict.setdefault(track_id, []).append(
                                    [*list(detection_int['abs_cartesian_x_y_z_smoothed']), *list(detection_int['velocity_x_y_z_smoothed'])]
                                    )
                                self.results[encounter_idx].append([row_idx, *list(detection_int['abs_cartesian_x_y_z_smoothed']), track_id, intruder_state['north_m'], intruder_state['east_m'], intruder_state['alt']])
                            track_ids.append(list(new_track_points.keys())[0])
                            #detection_confidences.append((list(new_track_points.values())[0]['detection_confidences'], list(new_track_points.values())[0]['horizon_statuses']))
                            # Will only be None if a daidalus call has not yet returned so we don't want to populate it
                            # if self.datastore['daidalus_out']['timestamp'] is not None:
                            #     daidalus_outputs.append([processed_ownship_state['trk'], response['suggested_hdir_deg'], response['horizontal_guidance'], list(new_track_points.values())[0]['coasting']])
                      
                        #########################################################################
                        ###### VISUALISATION OF MISSED DAIDALUS GUIDANCE DUE TO SATURATION ######
                        #########################################################################

                        # Extract guidance lists (may be empty)
                        guid_lists = [
                            response.get('horizontal_guidance', []),
                            response.get('altitude_guidance', []),
                            response.get('horizontal_speed_guidance', []),
                            response.get('vertical_speed_guidance', [])
                        ]

                        # Region types that are *not* considered conflicts
                        non_conflict_regions = {"NONE", "FAR"}
                        safe_regions = {"NONE", "FAR", "RECOVERY"}  # for "am I currently in a safe-ish band?"

                        # Condition:
                        # True if ANY list is non-empty AND all items in that list
                        # have region_type NOT in {"NONE", "FAR"}
                        # --- 1. Original guidance_conflict test: any list non-empty AND all entries are NOT NONE/FAR
                        guidance_conflict = any(
                            len(glist) > 0 and all(entry.get('region_type') not in non_conflict_regions
                                                for entry in glist)
                            for glist in guid_lists
                        )

                        # ---------- 2. Check if current state lies in ANY NONE/FAR/RECOVERY band ----------

                        def heading_in_band(angle_deg, lower_deg, upper_deg):
                            """Inclusive circular interval check [lower, upper] on [0,360)."""
                            if lower_deg is None or upper_deg is None:
                                return False

                            # Handle full-circle band
                            if abs((upper_deg - lower_deg) % 360) < 1e-9:
                                # lower=0, upper=360 or lower=10, upper=370 → full circle
                                return True

                            angle = angle_deg % 360.0
                            lo = lower_deg % 360.0
                            hi = upper_deg % 360.0

                            if lo <= hi:
                                return lo <= angle <= hi
                            else:
                                # wrap-around case, e.g. [350, 20]
                                return angle >= lo or angle <= hi

                        # current state (assumed to exist)
                        # trk   : current track in degrees [0,360)
                        # alt   : current altitude in meters
                        # h_vel : current horizontal speed in m/s (or whatever matches the JSON)
                        # v_vel : current vertical speed in m/s (or matching vertical band units)
                        trk = processed_ownship_state['trk']
                        alt = processed_ownship_state['alt']
                        h_vel = processed_ownship_state['ground_speed']
                        v_vel = processed_ownship_state['vertical_speed']

                        # Horizontal track safe?
                        in_safe_trk = any(
                            band.get('region_type') in safe_regions and
                            heading_in_band(trk,
                                            band.get('lower_trk_deg'),
                                            band.get('upper_trk_deg'))
                            for band in response.get('horizontal_guidance', [])
                        )

                        # Altitude safe?
                        in_safe_alt = any(
                            band.get('region_type') in safe_regions and
                            band.get('lower_alt_m') is not None and
                            band.get('upper_alt_m') is not None and
                            band['lower_alt_m'] <= alt <= band['upper_alt_m']
                            for band in response.get('altitude_guidance', [])
                        )

                        # Horizontal speed safe?
                        in_safe_hspd = any(
                            band.get('region_type') in safe_regions and
                            band.get('lower_h_spd_m_per_s') is not None and
                            band.get('upper_h_spd_m_per_s') is not None and
                            band['lower_h_spd_m_per_s'] <= h_vel <= band['upper_h_spd_m_per_s']
                            for band in response.get('horizontal_speed_guidance', [])
                        )

                        # Vertical speed safe?  (adjust key names to your JSON if needed)
                        in_safe_vspd = any(
                            band.get('region_type') in safe_regions and
                            band.get('lower_v_spd_m_per_s') is not None and
                            band.get('upper_v_spd_m_per_s') is not None and
                            band['lower_v_spd_m_per_s'] <= v_vel <= band['upper_v_spd_m_per_s']
                            for band in response.get('vertical_speed_guidance', [])
                        )

                        guidance_types = ["trk", "alt", "hspd", "vspd"]
                        oversaturated_guidance_types = [guidance_types[idx] for idx, x_type in enumerate([in_safe_trk, in_safe_alt, in_safe_hspd, in_safe_vspd]) if not x_type]

                        # If we're currently *inside* all safe/benign band, then we should not treat it as a guidance_conflict since no guidance expected
                        if guidance_conflict and not oversaturated_guidance_types:
                            guidance_conflict = False

                        ############ UNCOMMENT FOR DEBUGGING #############
                        # if guidance_conflict:
                        #     print("########### ALERT LEVEL!")
                        #     print(
                        #         (response['suggested_hdir_deg'], trk, in_safe_trk),
                        #         (response['suggested_hspd_knot']/1.944, h_vel, in_safe_hspd),
                        #         (response['suggested_vspd_fpm']/196.9, v_vel, in_safe_vspd),
                        #         (response['suggested_alt_ft']/3.281, alt, in_safe_alt),
                        #     )
                        #     print(response)

                        #########################################################################
                        #########################################################################
                        #########################################################################

                        #############################
                        ###### GUIDANCE LOGIC #######
                        #############################

                        guidance.append(np.nan)
                        targets_updated = False

                        if np.isfinite(response['suggested_hdir_deg']) or np.isfinite(response['suggested_hspd_knot']) or np.isfinite(response['suggested_vspd_fpm']) or np.isfinite(response['suggested_alt_ft']):
                            print("GUIDANCE RECIEVED!")
                            if first_guidance_idx is None:
                                first_guidance_idx = row_idx
                                dx = float(ownship_east_m[row_idx]) - float(intruder_east_m[row_idx])
                                dy = float(ownship_north_m[row_idx]) - float(intruder_north_m[row_idx])
                                dz = float(ownship_up_m[row_idx]) - float(intruder_up_m[row_idx])

                                hmd = float(np.sqrt(dx**2 + dy**2))
                                vmd = float(abs(dz))
                                guidance_info[f'encounter_{encounter_idx}'] = {
                                    'first_guidance_idx': first_guidance_idx,
                                    'ownship_time': ownship_time,
                                    'suggested_hdir_deg': response['suggested_hdir_deg'],
                                    'suggested_hspd_knot': response['suggested_hspd_knot'],
                                    'suggested_vspd_fpm': response['suggested_vspd_fpm'],
                                    'suggested_alt_ft': response['suggested_alt_ft'],
                                    'ownship_data': ownship_state,
                                    'intruder_data': intruder_state,
                                    'hmd_m': hmd,
                                    'vmd_m': vmd,
                                    'dx_m': dx,
                                    'dy_m': dy,
                                    'ownship_up_m': ownship_up_m[row_idx],
                                    'ownship_east_m': ownship_east_m[row_idx],
                                    'ownship_north_m': ownship_north_m[row_idx],
                                    'intruder_east_m': intruder_east_m[row_idx],
                                    'intruder_north_m': intruder_north_m[row_idx],
                                    'intruder_up_m': intruder_up_m[row_idx],
                                }

                            if response["intruders"]:
                                if response["intruders"][0]["alert_level"] == 2:
                                    self.statistics.log_alert(intruder_id=intruder_state['id'], alert_type='CA', timestamp=ownship_time)
                                if response["intruders"][0]["alert_level"] == 3:
                                    self.statistics.log_alert(intruder_id=intruder_state['id'], alert_type='WA', timestamp=ownship_time)

                            suggested_hdir_deg = response['suggested_hdir_deg'] if np.isfinite(response['suggested_hdir_deg']) else processed_ownship_state['trk']
                            suggested_hspd_mps = response['suggested_hspd_knot']/1.944 if np.isfinite(response['suggested_hspd_knot']) else processed_ownship_state['ground_speed']
                            suggested_vspd_mps = response['suggested_vspd_fpm']/196.9 if np.isfinite(response['suggested_vspd_fpm']) else processed_ownship_state['vertical_speed']
                            suggested_alt_m = response['suggested_alt_ft']/3.281 if np.isfinite(response['suggested_alt_ft']) else processed_ownship_state['alt']

                            guidance[-1] = response['suggested_hdir_deg'] if np.isfinite(response['suggested_hdir_deg']) else np.nan #suggested_hdir_deg
                            #guidance[-1] = suggested_hdir_deg

                            start_sim_idx = row_idx

                            if not self.start_simulating_ownship and start_sim_idx == row_idx:
                                
                                #print("Taking daidalus guidance and beginning ownship simulation")
                                self.start_simulating_ownship = True
                                #psi is yaw, phi is roll, theta is pitch
                                self.ownship_sim = SixDOFSim(processed_ownship_state['lat'], processed_ownship_state['lon'], processed_ownship_state['alt'], processed_ownship_state['trk'], processed_ownship_state['ground_speed'], processed_ownship_state['vertical_speed'], self.dt)
                                
                            
                            if self.start_simulating_ownship:
                                self.ownship_sim.update_targets(suggested_hdir_deg, suggested_hspd_mps, suggested_vspd_mps, suggested_alt_m)
                                targets_updated = True

                            # print("-------------")
                            # print(f"suggested heading: {suggested_hdir_deg}")
                            # print(f"suggested speed: {suggested_hspd_mps}")
                            # print(f"suggested vert speed: {suggested_vspd_mps}")
                            # print(f"suggested alt: {suggested_alt_m}")
                            # print("-------------")
                        else:
                            if guidance_conflict:
                                print(f"MISSING {oversaturated_guidance_types} GUIDANCE (OVERSATURATED)")
                                print(
                                    (response['suggested_hdir_deg'], trk, in_safe_trk),
                                    (response['suggested_hspd_knot']/1.944, h_vel, in_safe_hspd),
                                    (response['suggested_vspd_fpm']/196.9, v_vel, in_safe_vspd),
                                    (response['suggested_alt_ft']/3.281, alt, in_safe_alt),
                                )
                                print(response)

                                guidance[-1] = False

                            # Update iterations since last guidance recieved
                            if row_idx > 1:
                                for track_id_this in tracker.iso_decomp_guidance_hist:
                                    tracker.iso_decomp_guidance_hist[track_id_this]['iters_since_guidance'] += 1
                            self.results[encounter_idx].append([row_idx, np.nan, np.nan, np.nan, np.nan, intruder_state['north_m'], intruder_state['east_m'], intruder_state['alt']])

                        if self.start_simulating_ownship and not targets_updated:
                            self.ownship_sim.update_targets(processed_ownship_state['trk'], processed_ownship_state['ground_speed'], processed_ownship_state['vertical_speed'], processed_ownship_state['alt'])

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
                    guidance = np.array(guidance)

                    # import matplotlib.patches as mpatches

                    # # === Color map ===
                    # color_map = {
                    #     'NEAR': 'red',
                    #     'MID': 'orange',
                    #     'FAR': 'gray',
                    #     'RECOVERY': 'green',
                    #     'NONE': 'white'
                    # }

                    # # === Unpack data ===
                    # trk_values = []
                    # suggested_values = []
                    # region_data = []
                    # coasting_states = []

                    # for trk, suggested_hdir_deg, horizontal_guidance, coasting in daidalus_outputs:
                    #     trk_values.append(trk)
                    #     suggested_values.append(suggested_hdir_deg)
                    #     region_data.append(horizontal_guidance)  # expected to be list of dicts like [{"lower_trk_deg":..., "upper_trk_deg":..., "region_type":...}, ...]
                    #     coasting_states.append(coasting)

                    # # === Step 2: Plotting ===
                    # fig, ax = plt.subplots(figsize=(16, 6))

                    # # Plot the colored guidance bands
                    # for i, regions in enumerate(region_data):
                    #     for r in regions:
                    #         y0 = r['lower_trk_deg']
                    #         y1 = r['upper_trk_deg']
                    #         color = color_map.get(r['region_type'], 'black')
                    #         ax.plot([i, i], [y0, y1], color=color, linewidth=3)

                    # # Plot suggested headings (black dots)
                    # for i, val in enumerate(suggested_values):
                    #     if not np.isnan(val):
                    #         ax.plot(i, val, 'ko', label='Suggested Guidance' if i == 0 else "")

                    # # Plot current track, black when coasting=False, blue when coasting=True
                    # for i, (val, coast) in enumerate(zip(trk_values, coasting_states)):
                    #     if not np.isnan(val):
                    #         if coast:
                    #             ax.plot(i, val, 'bo', label='Current Heading (coasting)' if i == 0 else "")
                    #         else:
                    #             ax.plot(i, val, 'ko', label='Current Heading (tracking)' if i == 0 else "")

                    # # Legend
                    # handles = [mpatches.Patch(color=color_map[key], label=key) for key in color_map]
                    # handles.append(mpatches.Patch(color='black', label='Current Heading (tracking)'))
                    # handles.append(mpatches.Patch(color='blue', label='Current Heading (coasting)'))
                    # handles.append(mpatches.Patch(color='black', label='Suggested Guidance'))
                    # ax.legend(handles=handles, title="Region Type")

                    # ax.set_title("Guidance Bands Over Time")
                    # ax.set_xlabel("Time Index")
                    # ax.set_ylabel("Track Angle (degrees)")
                    # ax.set_ylim(0, 360)
                    # ax.grid(True)

                    # plt.tight_layout()

                    self.save_and_plot_encounter(encounter_idx, ownship_north_m, ownship_east_m, ownship_lla_array, intruder_lla_array_gt, intruder_lla_array_raw, intruder_lla_array_smoothed, intruder_lla_list_raw_dict, intruder_lla_list_smoothed_dict, track_ids, guidance, detection_confidences)
                    
                    self.statistics.reset()
                    self.intruder_track_id = 1

                    self.client.reset_daidalus(connection)

                    if self.async_guidance:
                        ge.stop()

                print(f"Total Stats for {ownship_file}, {intruder_file}:")
                print(f"Daidalus active: {self.use_daidalus}")
                stats = self.statistics.getTotalStats()
                stats['files'] = [ownship_file, intruder_file]
                self.save_summary_to_disk(stats)
                #self.plotter.save_all_encounters(ownship_file.split('/')[-2] + ' complete_encounters')
                print(stats)
                #guidance_info['files'] = [ownship_file, intruder_file]
                # output_path = f"results/guidance_info_{datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')}.json"
                # os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # with open(output_path, 'w') as f:
                #     json.dump(guidance_info, f, indent=4)

                # print(f"Guidance info saved to: {output_path}")

                # import csv 
                # # === Output CSV path ===
                # csv_filename = "encounter_results.csv"

                # # === Define headers ===
                # headers = ["Encounter_Id", "Frame", "X_m", "Y_m", "Z_m", "Track_Id", "Truth_X_m", "Truth_Y_m", "Truth_Z_m"]

                # # === Write to CSV ===
                # with open(csv_filename, mode="w", newline="") as f:
                #     writer = csv.writer(f)
                #     writer.writerow(headers)

                #     for encounter_id, rows in self.results.items():
                #         for row in rows:
                #             frame = row[0]
                #             x = row[1]
                #             y = row[2]
                #             z = row[3]
                #             track_id = row[4]
                #             truth_x = row[5]
                #             truth_y = row[6]
                #             truth_z = row[7]

                #             writer.writerow([encounter_id, frame, x, y, z, track_id, truth_x, truth_y, truth_z])

                # print(f"Saved to {csv_filename}")
                

if __name__ == '__main__':
    np.random.seed(3)
    ownship_file = './example_encounters/extracted_ownship_data73.mat'
    intruder_file = './example_encounters/extracted_intruder_data73.mat'

    # Mitigated (with Daidalus)
    main_mitigated = Main()
    main_mitigated.results = {}
    main_mitigated.use_daidalus = True
    main_mitigated.run(ownship_file, intruder_file, )#[14])#[7])#[19,7]) # Encoutners 0 and 4
    # mitigated_stats = main_mitigated.statistics
    # mitigated_summary = mitigated_stats.getTotalStats()
    # mitigated_summary['files'] = [ownship_file, intruder_file]
    # with open("results/mitigated_summary.json", "w") as f:
    #     json.dump(mitigated_summary, f, indent=2)

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

    # For the plot
    # np.random.seed(21)
    # ownship_file = './example_encounters/extracted_ownship_data73.mat'
    # intruder_file = './example_encounters/extracted_intruder_data73.mat'

    # # Mitigated (with Daidalus)
    # main_mitigated = Main()
    # main_mitigated.use_daidalus = True
    # main_mitigated.run(ownship_file, intruder_file, [19])