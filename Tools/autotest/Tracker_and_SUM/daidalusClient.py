import json
import time
import socket

class DaidalusClient():

    def __init__(self, socket_host, socket_port) -> None:

        # socket stuff
        self.socket_host = socket_host
        self.socket_port = socket_port

    def send_message(self, sock, message):
        message = message.encode()
        length = len(message)
        # print(f"sent {length} bytes")
        sock.sendall(length.to_bytes(4, byteorder='big'))  # Send message length
        sock.sendall(message)  # Send message

    def receive_message(self, sock):
        # First, receive the 4-byte length of the message
        raw_msglen = sock.recv(4)
        
        if not raw_msglen:
            return None
        
        # Unpack the message length
        msglen = int.from_bytes(raw_msglen, byteorder='big')
        
        # Receive the actual message
        data = sock.recv(msglen).decode('utf-8')
        return data
    
    def parse_response(self, response):
        if response:
            json_response = json.loads(response)
            # print("Received response from server:")
            # print(json.dumps(json_response, indent=4))
        else:
            json_response = None
            print("No response received from server.")

        return json_response

    def connect_to_daidalus_server(self):
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.socket_host, self.socket_port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                print(f"Connected to the Daidalus socket: {self.socket_host}:{self.socket_port}")
                return s
            except socket.error as e:
                self.current_ownship_pos = None
                print(f"Connection to Daidalus socket {self.socket_host}:{self.socket_port} failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)

    def reset_daidalus(self, connection):
        # send data to server
        final_data = {"reset": True}
        json_data = json.dumps(final_data)
        self.send_message(connection, json_data)

        # Receive response from server
        response = self.receive_message(connection)

    def send_data2(self, connection, final_data):

        json_data = json.dumps(final_data)
        # start = time.time()
        self.send_message(connection, json_data)
        # print(f"Send took: {(time.time() - start)*1000} ms")

        # Receive response from server
        # start = time.time()
        response = self.receive_message(connection)
        # print(f"network response took: {(time.time() - start)*1000} ms")
       
        # parse response for guidance info from Daidalus
        final_response = self.parse_response(response)

        return final_response

    def send_data(self, connection, ownship_position, intruder_positions):
        if ownship_position["coord_frame"] == "NEU":
            if ownship_position["centered"]:
                la_no = ownship_position["north_m_centred"]
                lo_ea = ownship_position["east_m_centred"]
            else:
                la_no = ownship_position["north_m"]
                lo_ea = ownship_position["east_m"]

            if ownship_position["aligned"]:
                ownship_position["coord_frame"] = "NEU_aligned"
        else:
            la_no = ownship_position["lat"]
            lo_ea = ownship_position["lon"]

        ownship_data = {"la_no": la_no,
                        "lo_ea": lo_ea,
                        "a": ownship_position["alt"],
                        "t": ownship_position["trk"],
                        "gs": ownship_position['ground_speed'],
                        "vs": ownship_position['vertical_speed'],
                        "ti": ownship_position['time'],
                        "cf": ownship_position["coord_frame"]}
        

        intruder_list = []

        for intruder_position in intruder_positions:

            if intruder_position["coord_frame"] == "NEU":
                if intruder_position["centered"]:
                    la_no = intruder_position["north_m_centred"]
                    lo_ea = intruder_position["east_m_centred"]
                else:
                    la_no = intruder_position["north_m"]
                    lo_ea = intruder_position["east_m"]

                if intruder_position["aligned"]:
                    trk = intruder_position["trk_aligned"]
                else:
                    trk = intruder_position["trk"]
            else:
                la_no = intruder_position["lat"]
                lo_ea = intruder_position["lon"]
                trk = intruder_position["trk"]

            intruder = {"la_no": la_no,
                        "lo_ea": lo_ea,
                        "a": intruder_position["alt"],
                        "t": trk,
                        "gs": intruder_position['ground_speed'],
                        "vs": intruder_position['vertical_speed'],
                        "ty": intruder_position['type'],
                        "c": intruder_position['callsign'],
                        "i": intruder_position['id'],
                        "ti": intruder_position['time'],
                        "cf": intruder_position["coord_frame"],
                        "range_det": intruder_position['range_uncertainty']}
                    
            intruder_list.append(intruder)
                
        # send data to server
        final_data = {"o": ownship_data,
                        "d": intruder_list}

        json_data = json.dumps(final_data)
        # start = time.time()
        self.send_message(connection, json_data)
        # print(f"Send took: {(time.time() - start)*1000} ms")

        # Receive response from server
        # start = time.time()
        response = self.receive_message(connection)
        # print(f"network response took: {(time.time() - start)*1000} ms")
       
        # parse response for guidance info from Daidalus
        final_response = self.parse_response(response)

        return final_response