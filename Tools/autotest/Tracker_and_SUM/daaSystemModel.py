from Tracker_and_SUM import airspaceStatistics
import math

class DAASystemModel():

    def __init__(self) -> None:
        self.FOR_horizontal_deg = 270
        self.FOR_vertical_deg = 60
        self.max_detection_range_m = 3000

    def check_FOR_Range(self, ownship_pos, intruder_pos):

        ownship_lat, ownship_lon, ownship_alt, ownship_heading = ownship_pos['lat'], ownship_pos['lon'], ownship_pos['alt'], ownship_pos['trk']

        intruder_lat, intruder_lon, intruder_alt = intruder_pos['lat'], intruder_pos['lon'], intruder_pos['alt']

        # Convert lat/lon to radians
        camera_lat_rad = math.radians(ownship_lat)
        camera_lon_rad = math.radians(ownship_lon)
        object_lat_rad = math.radians(intruder_lat)
        object_lon_rad = math.radians(intruder_lon)

        # Earth's radius in meters
        R = 6371000

        # Calculate distance using the Haversine formula
        delta_lat = object_lat_rad - camera_lat_rad
        delta_lon = object_lon_rad - camera_lon_rad

        a = math.sin(delta_lat / 2)**2 + math.cos(camera_lat_rad) * math.cos(object_lat_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        # Calculate bearing (horizontal angle)
        bearing = math.degrees(math.atan2(math.sin(delta_lon) * math.cos(object_lat_rad),
                                        math.cos(camera_lat_rad) * math.sin(object_lat_rad) - 
                                        math.sin(camera_lat_rad) * math.cos(object_lat_rad) * math.cos(delta_lon)))
        
        # Calculate relative bearing by adjusting with the camera heading
        relative_bearing = (bearing - ownship_heading + 360) % 360
        if relative_bearing > 180:  # Normalize to [-180, 180] range
            relative_bearing -= 360

        # Calculate vertical angle (elevation)
        altitude_difference = intruder_alt - ownship_alt
        vertical_angle = math.degrees(math.atan2(altitude_difference, distance))

        # Check if within FOV and range
        if distance <= self.max_detection_range_m:
            if (-self.FOR_horizontal_deg / 2 <= relative_bearing <= self.FOR_horizontal_deg / 2 and
                -self.FOR_vertical_deg / 2 <= vertical_angle <= self.FOR_vertical_deg / 2):
                # print("Object is within view and range.")
                return True
            else:
                # print("Object is within range but out of FOV.")
                return False
        else:
            # print("Object is out of range.")
            return False

