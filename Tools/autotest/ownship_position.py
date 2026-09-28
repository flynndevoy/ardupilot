class OwnshipPosition():
    def __init__(self, global_pos_dict, time_recv):
        self.lat = int(global_pos_dict['lat'])/10000000.0 #degE7 to deg
        self.lon = int(global_pos_dict['lon'])/10000000.0 #degE7 to deg
        self.alt = int(global_pos_dict['alt'])/1000.0 #mm to m (ASL/MSL)
        self.rel_alt = int(global_pos_dict['relative_alt'])/1000.0 #mm to m (AGL)
        self.hdg = int(global_pos_dict['hdg'])/100.0 #cdeg to deg
        self.vx = int(global_pos_dict['vx'])/100.0 #cm/s to m/s
        self.vy = int(global_pos_dict['vy'])/100.0 #cm/s to m/s
        self.vz = -1*int(global_pos_dict['vz'])/100.0 #cm/s to m/s, Ardupilot reports Positive = DOWN (Converted to Positive = UP)
        self.time_boot_ms = int(global_pos_dict['time_boot_ms']) #ms
        self.time_recv = time_recv