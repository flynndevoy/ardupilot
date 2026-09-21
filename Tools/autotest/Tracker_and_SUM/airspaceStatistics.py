import numpy as np

# Constants (distances are in meters)
ENCOUNTER_HORIZONTAL_THRESHOLD = 915    #3000ft
ENCOUNTER_VERTICAL_THRESHOLD = 107  #350ft 

WELL_CLEAR_HORIZONTAL_THRESHOLD = 610   #2000ft
WELL_CLEAR_VERTICAL_THRESHOLD = 77  #250ft

NMAC_HORIZONTAL_THRESHOLD = 153 #500ft
NMAC_VERTICAL_THRESHOLD = 31    #100ft

MAC_HORIZONTAL_THRESHOLD = 10          # Mid-Air Collision if within 10m
MAC_VERTICAL_THRESHOLD = 10          # Mid-Air Collision if within 10m

VERTICAL_RATE_THRESHOLD = 10.16         # ~2000 feet per minute in m/s
HORIZONTAL_RATE_THRESHOLD = 105.4        # ~205 knots in m/s

# Radius of the Earth in meters
EARTH_RADIUS = 6371000

class AirspaceStatistics():

    def __init__(self) -> None:
        # self.results = []
        self.reset()

        self.total_encounters = 0
        self.total_wcv = 0
        self.total_nmacs = 0
        self.total_macs = 0
        self.total_two_or_fewer = 0
        self.total_three_or_four = 0
        self.total_three_no_nmac = 0
        self.total_ca_count = 0
        self.total_c21_count = 0
        self.total_c20_count = 0

    def reset(self):
        self.wcv_active = False
        self.nmac_active = False
        self.mac_active = False
        self.encounter_count = 0
        self.wcv_count = 0
        self.nmac_count = 0
        self.mac_count = 0
        self.two_or_fewer = 0
        self.three_or_four = 0
        self.three_no_nmac = 0

        self.ca_count = 0
        self.c21_count = 0
        self.c20_count = 0

        self.closest_horizontal = float('inf')
        self.closest_vertical = float('inf')
        self.closest_id_horizontal = -1
        self.closest_id_vertical = -1

        self.flight_history = {}
        self.alert_log = {}
        self.severity_tagged_intruders = set()
        
        self.encounter_ids_count = {}
        self.wcv_ids_count = {}
        self.nmac_ids_count = {}
        self.mac_ids_count = {}

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the great-circle distance between two points on the Earth (specified in decimal degrees)."""
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distance = EARTH_RADIUS * c  # Distance in meters
        return distance
    
    def calculate_vertical_distance(self, alt1, alt2):
        """Calculate the vertical distance between two altitudes."""
        return abs(alt1 - alt2)
    
    def finishEncounter(self):

        # sum total encounters for each intruder in the sim
        for val in self.encounter_ids_count.values():
            self.encounter_count+=val['encounter_count']

            # alternative way to count encounters if you don't want to include multiple encounters from the same intruder
            # if val['encountered']:
                #self.encounter_count+=1

        self.total_encounters+=self.encounter_count

        for val in self.wcv_ids_count.values():
            self.wcv_count+=val['wcv_count']

        self.total_wcv+=self.wcv_count

        for val in self.nmac_ids_count.values():
            self.nmac_count+=val['nmac_count']

        self.total_nmacs+=self.nmac_count

        for val in self.mac_ids_count.values():
            self.mac_count+=val['mac_count']

        self.total_macs+=self.mac_count

        self.total_two_or_fewer += self.two_or_fewer
        self.total_three_or_four += self.three_or_four
        self.total_three_no_nmac += self.three_no_nmac

        # Process alert log for C20 / C21
        for intruder_id, alerts in self.alert_log.items():
            ca_times = [a['time'] for a in alerts if a['type'] == 'CA']
            wa_times = [a['time'] for a in alerts if a['type'] == 'WA']

            for ca_time in ca_times:
                # Find all WA within 14 seconds after this CA
                deltas = [wa_time - ca_time for wa_time in wa_times if wa_time >= ca_time]
                if not deltas:
                    continue
                min_delta = min(deltas)
                if min_delta < 1.0:
                    self.c21_count += 1
                elif min_delta < 14.0:
                    self.c20_count += 1

            self.ca_count += len(ca_times)

        self.total_ca_count += self.ca_count
        self.total_c20_count += self.c20_count
        self.total_c21_count += self.c21_count

        
        return {
            'total_encounters': self.encounter_count,
            'enc_by_id': self.encounter_ids_count,
            'total_well_clear_violations': self.wcv_count,
            'wcv_by_id': self.wcv_ids_count,
            'total_nmacs': self.nmac_count,
            'nmacs_by_id': self.nmac_ids_count,
            'total_macs': self.mac_count,
            'macs_by_id': self.mac_ids_count,
            'total_two_or_fewer': self.two_or_fewer,
            'total_three_or_four': self.three_or_four,
            'total_three_no_nmac': self.three_no_nmac,
            'c21_count': self.c21_count,
            'c20_count': self.c20_count,
            'ca_count': self.ca_count,
            'closest_horizontal_approach': self.closest_horizontal,
            'closest_vertical_approach': self.closest_vertical,
            'closest_horizontal_approach_id': self.closest_id_horizontal,
            'closest_vertical_approach_id': self.closest_id_vertical
        }
    
    def getTotalStats(self):
         return {
            'total_encounters': self.total_encounters,
            'total_wcv': self.total_wcv,
            'total_nmacs': self.total_nmacs,
            'total_macs': self.total_macs,
            'total_two_or_fewer': self.total_two_or_fewer,
            'total_three_or_four': self.total_three_or_four,
            'total_three_no_nmac': self.total_three_no_nmac,
            'total_ca_count': self.total_ca_count,
            'total_c20_count': self.total_c20_count,
            'total_c21_count': self.total_c21_count
        }
    
    def log_alert(self, intruder_id, alert_type, timestamp):
        if intruder_id not in self.alert_log:
            self.alert_log[intruder_id] = []
        self.alert_log[intruder_id].append({
            'type': alert_type,
            'time': timestamp
        })

    def evaluate(self, ownship_pos, intruder_pos_list):
        """Evaluate between the ownship and a list of intruders."""
        lat1, lon1, alt1 = ownship_pos['lat'], ownship_pos['lon'], ownship_pos['alt']
        trk1 = ownship_pos['trk']
        gs1 = ownship_pos['ground_speed']
        vs1 = ownship_pos['vertical_speed']
        own_time = ownship_pos['time']

        results = {}

        for intruder in intruder_pos_list:
            lat2, lon2, alt2 = intruder['lat'], intruder['lon'], intruder['alt']
            trk2 = intruder['trk']
            gs2 = intruder['ground_speed']
            vs2 = intruder['vertical_speed']

            horizontal_distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            vertical_distance = self.calculate_vertical_distance(alt1, alt2)

            if horizontal_distance < self.closest_horizontal:
                self.closest_horizontal = horizontal_distance
                self.closest_id_horizontal = intruder['id']
            
            if vertical_distance < self.closest_vertical:
                self.closest_vertical = vertical_distance
                self.closest_id_vertical = intruder['id']
        
            encounter = (horizontal_distance <= ENCOUNTER_HORIZONTAL_THRESHOLD and vertical_distance <= ENCOUNTER_VERTICAL_THRESHOLD)
            wcv = (horizontal_distance <= WELL_CLEAR_HORIZONTAL_THRESHOLD and vertical_distance <= WELL_CLEAR_VERTICAL_THRESHOLD)
            nmac = (horizontal_distance <= NMAC_HORIZONTAL_THRESHOLD and vertical_distance <= NMAC_VERTICAL_THRESHOLD)
            mac = (horizontal_distance <= MAC_HORIZONTAL_THRESHOLD and vertical_distance <= MAC_VERTICAL_THRESHOLD)

            # Calculate severity
            severity_criteria = 0
            if abs(vs1 - vs2) >= VERTICAL_RATE_THRESHOLD:
                severity_criteria += 1
            if abs(gs1 - gs2) >= HORIZONTAL_RATE_THRESHOLD:
                severity_criteria += 1

            # Only tag each intruder once for severity count
            if wcv and intruder['id'] not in self.severity_tagged_intruders:
                if severity_criteria <= 1:
                    self.two_or_fewer += 1
                else:
                    self.three_or_four += 1
                    if not nmac:
                        self.three_no_nmac += 1
                self.severity_tagged_intruders.add(intruder['id'])

                
            # Handle Encounter
            if encounter:
                # Starting a new encounter with a new intruder
                if intruder['id'] not in self.encounter_ids_count.keys():
                    self.encounter_ids_count[intruder['id']] = {"encounter_active": True, "encounter_count": 1, 'encountered': True}
                # intruder has been re-encountered
                elif not self.encounter_ids_count[intruder['id']]["encounter_active"]:
                    self.encounter_ids_count[intruder['id']]["encounter_active"] = True
                    self.encounter_ids_count[intruder['id']]["encounter_count"]+=1
            # No ongoing encounter with intruder, reset encounter flag
            else:
                if intruder['id'] in self.encounter_ids_count.keys():
                    self.encounter_ids_count[intruder['id']]["encounter_active"] = False

            # Handle Well Clear Violation (WCV)
            if wcv:
                if intruder['id'] not in self.wcv_ids_count.keys():
                    self.wcv_ids_count[intruder['id']] = {"wcv_active": True, "wcv_count": 1, 'well_clear_violated': True}
                elif not self.wcv_ids_count[intruder['id']]["wcv_active"]:
                    self.wcv_ids_count[intruder['id']]["wcv_active"] = True
                    self.wcv_ids_count[intruder['id']]["wcv_count"] += 1
            else:
                if intruder['id'] in self.wcv_ids_count.keys():
                    self.wcv_ids_count[intruder['id']]["wcv_active"] = False

            # Handle Near Mid-Air Collision (NMAC)
            if nmac:
                if intruder['id'] not in self.nmac_ids_count.keys():
                    self.nmac_ids_count[intruder['id']] = {"nmac_active": True, "nmac_count": 1, 'nmac_violated': True}
                elif not self.nmac_ids_count[intruder['id']]["nmac_active"]:
                    self.nmac_ids_count[intruder['id']]["nmac_active"] = True
                    self.nmac_ids_count[intruder['id']]["nmac_count"] += 1
            else:
                if intruder['id'] in self.nmac_ids_count.keys():
                    self.nmac_ids_count[intruder['id']]["nmac_active"] = False

            # Handle Mid-Air Collision (MAC)
            if mac:
                if intruder['id'] not in self.mac_ids_count.keys():
                    self.mac_ids_count[intruder['id']] = {"mac_active": True, "mac_count": 1, 'mac_violated': True}
                elif not self.mac_ids_count[intruder['id']]["mac_active"]:
                    self.mac_ids_count[intruder['id']]["mac_active"] = True
                    self.mac_ids_count[intruder['id']]["mac_count"] += 1
            else:
                if intruder['id'] in self.mac_ids_count.keys():
                    self.mac_ids_count[intruder['id']]["mac_active"] = False

            results[intruder['id']] = {
                'encounter': encounter,
                'wcv': wcv,
                'nmac': nmac,
                'mac': mac,
                'horizontal_distance_m': horizontal_distance,
                'vertical_distance_m': vertical_distance,
                'severity_criteria': severity_criteria
            }

            if intruder['id'] not in self.flight_history.keys():
                self.flight_history[intruder['id']] = []
                self.flight_history[intruder['id']].append(results[intruder['id']])
            else:
                self.flight_history[intruder['id']].append(results[intruder['id']])

        return results

    @staticmethod
    def compare_runs(baseline, mitigated):
        wc_unresolved = 0
        wc_induced = 0
        nmac_unresolved = 0
        nmac_induced = 0
        two_or_fewer_unresolved = 0
        three_or_four_unresolved = 0
        nmac_unresolved_count = 0

        # WCV Comparison
        baseline_wcv = set(baseline.wcv_ids_count.keys())
        mitigated_wcv = set(mitigated.wcv_ids_count.keys())

        wc_unresolved_ids = baseline_wcv & mitigated_wcv
        wc_induced_ids = mitigated_wcv - baseline_wcv

        wc_unresolved = len(wc_unresolved_ids)
        wc_induced = len(wc_induced_ids)

        # NMAC Comparison
        baseline_nmac = set(baseline.nmac_ids_count.keys())
        mitigated_nmac = set(mitigated.nmac_ids_count.keys())

        nmac_unresolved_ids = baseline_nmac & mitigated_nmac
        nmac_induced_ids = mitigated_nmac - baseline_nmac

        nmac_unresolved = len(nmac_unresolved_ids)
        nmac_induced = len(nmac_induced_ids)

        # Detailed Unresolved Breakdown (severity)
        for intruder_id in wc_unresolved_ids:
            if intruder_id in baseline.severity_tagged_intruders:
                if intruder_id in baseline.wcv_ids_count:
                    criteria = baseline.flight_history[intruder_id][0]['severity_criteria']
                    if criteria <= 1:
                        two_or_fewer_unresolved += 1
                    else:
                        three_or_four_unresolved += 1

        nmac_unresolved_count = len(nmac_unresolved_ids)

        return {
            'wc_unresolved': wc_unresolved,
            'wc_induced': wc_induced,
            'nmac_unresolved': nmac_unresolved,
            'nmac_induced': nmac_induced,
            'two_or_fewer_unresolved': two_or_fewer_unresolved,
            'three_or_four_unresolved': three_or_four_unresolved,
            'nmac_unresolved_count': nmac_unresolved_count
        }