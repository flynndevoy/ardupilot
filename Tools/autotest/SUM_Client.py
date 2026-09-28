import json
import time
import socket
import numpy as np
# from prediction_structure_v2 import DetectedItem
from Tracker_and_SUM.prediction_structure_v2 import DetectedItem

# 1) A custom JSON encoder
class DetectionJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # NumPy array -> list + marker
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": obj.tolist()}
        # NumPy scalar -> Python native
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        # Your custom class -> dict + marker
        if isinstance(obj, DetectedItem):
            data = obj.__dict__.copy()
            data["__DetectedItem__"] = True
            return data
        # Let the base class raise the TypeError
        return super().default(obj)

class SUMClient():
    """
    A client to connect to the SUM (Sensor Uncertainty Model) server, send intruder data,
    and receive responses.
    """

    def __init__(self, socket_host, socket_port) -> None:
        """
        Initializes the client with the server's host and port.

        Args:
            socket_host (str): The hostname or IP address of the server.
            socket_port (int): The port number of the server.
        """
        # socket stuff
        self.socket_host = socket_host
        self.socket_port = socket_port

    def send_message(self, sock, message):
        """
        Encodes and sends a message to the given socket.
        The message is prefixed with a 4-byte length header to ensure
        the server knows exactly how much data to read.

        Args:
            sock (socket.socket): The connected socket to send the message through.
            message (str): The JSON string message to send.
        """
        # Encode the string message into bytes using UTF-8
        message = message.encode('utf-8')
        # Get the length of the encoded message
        length = len(message)
        # Send the message length as a 4-byte integer in big-endian format
        sock.sendall(length.to_bytes(4, byteorder='big'))
        # Send the actual message
        sock.sendall(message)

    def receive_message(self, sock):
        """
        Receives a message from the given socket.
        It first reads a 4-byte length header to determine the message size.

        Args:
            sock (socket.socket): The connected socket to receive the message from.

        Returns:
            str: The decoded message string, or None if the server closes the connection.
        """
        # First, receive the 4-byte length of the incoming message
        raw_msglen = sock.recv(4)

        # If recv returns an empty string, it means the other side closed the connection
        if not raw_msglen:
            return None

        # Unpack the message length from big-endian format
        msglen = int.from_bytes(raw_msglen, byteorder='big')

        # Receive the actual message data until the full message is received
        data = sock.recv(msglen).decode('utf-8')
        return data

    def parse_response(self, response):
        """
        Parses the JSON response from the server.

        Args:
            response (str): The JSON string received from the server.

        Returns:
            dict: The parsed JSON response as a Python dictionary, or None if the response is empty.
        """
        if response:
            # Load the JSON string into a Python dictionary
            json_response = json.loads(response)
            # You can uncomment these lines for debugging purposes
            # print("Received response from server:")
            # print(json.dumps(json_response, indent=4))
        else:
            json_response = None
            print("No response received from server.")

        return json_response

    def connect_to_SUM_server(self):
        """
        Attempts to connect to the SUM server in a loop until successful.
        This makes the client resilient to the server not being available immediately.

        Returns:
            socket.socket: The connected socket object.
        """
        while True:
            try:
                # Create a new TCP/IP socket
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # Attempt to connect to the server
                s.connect((self.socket_host, self.socket_port))
                print(f"Connected to the SUM socket: {self.socket_host}:{self.socket_port}")
                return s
            except socket.error as e:
                # If connection fails, print an error and wait before retrying
                print(f"Connection to SUM socket {self.socket_host}:{self.socket_port} failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)

    def send_detection(self, connection, synthetic_detection, ownship_position, ownship_attitude):
        """
        Formats intruder data, sends it to the server, and awaits a response.

        Args:
            connection (socket.socket): The connected socket to the server.


        Returns:
            dict: The parsed JSON response from the server.
        """

        # Prepare the final data structure to be sent
        final_data = {
            "synthetic_detection": synthetic_detection,
            "ownship_position": ownship_position,
            "ownship_attitude": ownship_attitude,
        }
        # Convert the dictionary to a JSON string
        json_data = json.dumps(final_data, cls=DetectionJsonEncoder)

        # Send the JSON data to the server
        try:
            self.send_message(connection, json_data)
        except Exception as e:
            print(f"Intruder message via SUM Client failed: {e}")
            return None

        # Receive the response from the server
        response = self.receive_message(connection)

        # Parse the JSON response
        final_response = self.parse_response(response)

        return final_response


    def send_daidalus_data(self, connection, daidalus_input):
        """
        Formats intruder data, sends it to the server, and awaits a response.

        Args:
            connection (socket.socket): The connected socket to the server.
            daidalus_input (dict): A dictionary containing ownship (dict) and intruder data (list<dict>)

        Returns:
            dict: The parsed JSON response from the server.
        """
        # Convert the dictionary to a JSON string
        json_data = json.dumps(daidalus_input)

        # Send the JSON data to the server
        try:
            self.send_message(connection, json_data)
        except Exception as e:
            print(f"Intruder message via SUM Client failed: {e}")
            return None

        # Receive the response from the server
        response = self.receive_message(connection)

        # Parse the JSON response
        final_response = self.parse_response(response)

        return final_response


if __name__ == '__main__':
    # --- Configuration ---
    # This should match the host and port the server is running on.
    # '127.0.0.1' (or 'localhost') is used for running client and server on the same machine.
    SERVER_HOST = '127.0.0.1'
    SERVER_PORT = 65434

    # --- Main Execution ---
    print("Starting SUM Client...")
    # 1. Create an instance of the client
    client = SUMClient(SERVER_HOST, SERVER_PORT)

    # 2. Connect to the server
    server_connection = client.connect_to_SUM_server()

    synthetic_detection_0 = {
        'curr_frames': [None, None, None, None, None, None],
        'prev_frames': [None, None, None, None, None, None],
        'predicted_transforms': [
            np.array(
                [[ 1.00185338e+00, -4.64397768e-03,  7.64119488e+00],
                [-1.46022121e-06,  1.00370226e+00, -1.52918069e+01],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            ), np.array(
                [[ 9.99797349e-01, -9.10405050e-03,  3.28120711e+01],
                [-9.41342493e-08,  9.99553770e-01,  1.83415978e+00],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            ), np.array(
                [[ 9.97212111e-01, -2.05841148e-03,  2.02196796e+01],
                [ 1.13942929e-06,  9.94433383e-01,  2.29679027e+01],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            ), np.array(
                [[ 1.00185795e+00,  4.61900720e-03, -2.48819975e+01],
                [ 1.90613053e-06,  1.00370974e+00, -1.53379723e+01],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            ), np.array(
                [[ 9.99804864e-01,  9.10661915e-03, -3.09468985e+01],
                [ 3.24200663e-07,  9.99569132e-01,  1.76884585e+00],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            ), np.array(
                [[ 9.97213590e-01,  2.08650165e-03,  5.73337286e+00],
                [-5.10417536e-07,  9.94437184e-01,  2.29597585e+01],
                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
            )
        ],
        'detections': [
            [],
            [DetectedItem(
                cx=np.float64(5475.416703885544),
                cy=np.float64(3318.025988136198),
                w=100,
                h=100,
                item_id='',
                distance=np.float64(1396.632527939226),
                confidence=0.8,
                dx=np.float64(-8.884697132412839),
                dy=np.float64(0.6517705133742311),
                abs_cx=np.float64(5475.416703885544),
                abs_cy=np.float64(3318.025988136198),
                above_horizon=0.0,
                cls_idx=0,
                primary_conf=0.8,
                cam_id=1,
                is_matched_planned=False,
                is_matched_unplanned=False,
                matched_item_id='',
                track_id=-1,
                add_to_submit=False
            )],
            [],
            [],
            [],
            []
        ],
        'tracks': [None, None, None, None, None, None],
        'frame_id': ['0_36.900000000000006', '1_36.900000000000006', '2_36.900000000000006', '3_36.900000000000006', '4_36.900000000000006', '5_36.900000000000006'],
        'prev_frames_aligned_affine': [None, None, None, None, None, None],
        'total_detect_track_duration': [None, None, None, None, None, None]
    }

    ownship_position_0 = {
        'lat_deg': -27.266337, # Position in degrees
        'lon_deg': 152.515640,
        'alt_msl_m': 1000.0, # Altitude in meters
        'relative_home_alt_m': 1000.0,
        'ground_vx_m_s': 40.0 * np.cos(np.radians(334.8)), # Speed in m/s
        'ground_vy_m_s': 40.0 * np.sin(np.radians(334.8)),
        'ground_vz_m_s': 0.0,
        'hdg_deg': 334.8, # Heading in degrees
        'timestamp': 36.9
    }

    ownship_attitude_0 = {
        'timestamp': 36.9,
        'roll_deg': 0,
        'pitch_deg': 0,
        'yaw_deg': 0,
        'roll_rate_deg_s': 0,
        'pitch_rate_deg_s': 0,
        'yaw_rate_deg_s': 0
    }

    daidalus_input_0 = {
        'o': {
            'la_no': np.float64(62.853255964249996),
            'lo_ea': np.float64(-24.958625645019644),
            'a': 687.99,
            't': 338.1,
            'gs': np.float64(50.43136424884816),
            'vs': 1.25,
            'ti': 1.5,
            'cf': 'NEU'
        },
        'd': [{
            'la_no': np.float64(3280.102848525293),
            'lo_ea': np.float64(708.6092639737775),
            'a': np.float64(736.6331028067586),
            't': np.float64(279.02155669104616),
            'gs': np.float64(26.104411357858258),
            'vs': np.float64(1.77151314545674),
            'c': 'INTRUDER',
            'i': 1,
            'ty': 'DAA',
            'ti': np.float64(1.5),
            'cf': 'NEU'
        }]
    }

    try:
        # 3. Run a loop to send data periodically
        i = 0
        while True:
            # Create some sample intruder data. In a real application,
            # this data would come from a sensor or another source.
            imod = i % 2

            print(f"\nSending data for detections...")

            # Send the data and get the response
            if imod == 0:
                print("sending detection")
                server_response = client.send_detection(server_connection, synthetic_detection_0, ownship_position_0, ownship_attitude_0)
            else:
                print("sending daidalus")
                server_response = client.send_daidalus_data(server_connection, daidalus_input_0)

            if server_response:
                print(f"Received from server: {server_response}")

            # Wait for 5 seconds before sending the next update
            time.sleep(1)
            i += 1

    except KeyboardInterrupt:
        # This allows you to stop the client gracefully with Ctrl+C
        print("\nClient shutting down.")
    except socket.error as e:
        # Handle cases where the server might close the connection unexpectedly
        print(f"Socket error: {e}. Connection lost.")
    finally:
        # Ensure the connection is closed when the loop exits
        if server_connection:
            server_connection.close()
            print("Connection to server closed.")
