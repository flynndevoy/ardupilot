# To generate grpc files:
# python3 -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. *.proto

import threading
import time
import io
import grpc
from concurrent import futures

import msg_to_hmi_pb2
import msg_to_hmi_pb2_grpc
import statuses_updates_pb2
import statuses_updates_pb2_grpc
import camera_stream_pb2
import camera_stream_pb2_grpc

TARGET_UPDATE_FREQ = 5 #hz
TARGET_WAIT_TIME = 1.0/TARGET_UPDATE_FREQ

ownship_data = None
traffic_data = None
daidalus_output_data = None
system_status_data = None
controller_mode_data = None

flight_data_lock = threading.Lock()

class FlightDataService(msg_to_hmi_pb2_grpc.FlightDataServiceServicer):
    def SubscribeFlightData(self, request, context):

        global ownship_data, traffic_data, daidalus_output_data, system_status_data, controller_mode_data

        print(f"[FlightData] Subscription started for session: {request.session_id}")

        while True:

            start = time.time()

            with flight_data_lock:

                if ownship_data is not None and traffic_data is not None and daidalus_output_data is not None and system_status_data is not None and controller_mode_data is not None:

                    ownship = msg_to_hmi_pb2.OwnshipMsg(
                        id=ownship_data['id'],
                        is_active=ownship_data['is_active'],
                        s=msg_to_hmi_pb2.Coordinates(
                            lat=ownship_data['s']['lat'],
                            lon=ownship_data['s']['lon'],
                            alt=ownship_data['s']['alt']
                        ),
                        v=msg_to_hmi_pb2.Velocity(
                            x=ownship_data['v']['x'],
                            y=ownship_data['v']['y'],
                            z=ownship_data['v']['z']
                        ),
                        wind=msg_to_hmi_pb2.Wind(
                            speed=ownship_data['wind']['speed'],
                            direction=ownship_data['wind']['direction']
                        )
                    )

                    traffic = [
                        msg_to_hmi_pb2.TrafficMsg(
                            id=t['id'],
                            is_active=t['is_active'],
                            s=msg_to_hmi_pb2.Coordinates(
                                lat=t['s']['lat'],
                                lon=t['s']['lon'],
                                alt=t['s']['alt']
                            ),
                            v=msg_to_hmi_pb2.Velocity(
                                x=t['v']['x'],
                                y=t['v']['y'],
                                z=t['v']['z']
                            ),
                            t_type=t['t_type'],
                            bounding_box=msg_to_hmi_pb2.ImageFrame(
                                image_data=t['bounding_box']['image_data'],
                                timestamp=t['bounding_box']['timestamp']
                            )
                        ) for t in traffic_data
                    ]

                    daidalus = msg_to_hmi_pb2.DaidalusOutput(
                        horizontal_guidance=[
                            msg_to_hmi_pb2.HorizontalGuidance(
                                lower_trk_deg=hg['lower_trk_deg'],
                                upper_trk_deg=hg['upper_trk_deg'],
                                region_type=hg['region_type']
                            ) for hg in daidalus_output_data['horizontal_guidance']
                        ],
                        altitude_guidance=[
                            msg_to_hmi_pb2.AltitudeGuidance(
                                lower_alt_m=ag['lower_alt_m'],
                                upper_alt_m=ag['upper_alt_m'],
                                region_type=ag['region_type']
                            ) for ag in daidalus_output_data['altitude_guidance']
                        ],
                        horizontal_speed_guidance=[
                            msg_to_hmi_pb2.HorizontalSpeedGuidance(
                                lower_h_spd_m_per_s=hsg['lower_h_spd_m_per_s'],
                                upper_h_spd_m_per_s=hsg['upper_h_spd_m_per_s'],
                                region_type=hsg['region_type']
                            ) for hsg in daidalus_output_data['horizontal_speed_guidance']
                        ],
                        vertical_speed_guidance=[
                            msg_to_hmi_pb2.VerticalSpeedGuidance(
                                lower_v_spd_m_per_s=vsg['lower_v_spd_m_per_s'],
                                upper_v_spd_m_per_s=vsg['upper_v_spd_m_per_s'],
                                region_type=vsg['region_type']
                            ) for vsg in daidalus_output_data['vertical_speed_guidance']
                        ],
                        intruders=[
                            msg_to_hmi_pb2.IntruderAlert(
                                id=ia['id'],
                                t2los_sec=ia.get('t2los_sec'),
                                t2cpa_sec=ia.get('t2cpa_sec'),
                                horizontal_separation_NM=ia.get('horizontal_separation_NM'),
                                alert_level=ia['alert_level']
                            ) for ia in daidalus_output_data['intruders']
                        ],
                        suggested_hdir_deg=daidalus_output_data.get('suggested_hdir_deg'),
                        suggested_hspd_knot=daidalus_output_data.get('suggested_hspd_knot'),
                        suggested_vspd_fpm=daidalus_output_data.get('suggested_vspd_fpm'),
                        suggested_alt_ft=daidalus_output_data.get('suggested_alt_ft')
                    )


                    controller_mode = {
                        "mode": controller_mode_data['mode']
                    }

                    # prepare message
                    response = msg_to_hmi_pb2.FlightDataStreamResponse(
                        ownship_msg=ownship,
                        traffic=traffic,
                        daidalus_output=daidalus,
                        controller_mode = controller_mode,
                        )

                    yield response

                    ownship_data = None  # Reset after sending
                    traffic_data = None  # Reset after sending
                    daidalus_output_data = None  # Reset after sending
                    system_status_data = None  # Reset after sending
                    controller_mode_data = None  # Reset after sending

            # control loop freq
            time.sleep(max(TARGET_WAIT_TIME - (time.time() - start), 0))

class DaaStatusService(statuses_updates_pb2_grpc.DaaStatusServiceServicer):
    def SubscribeToStatusUpdates(self, request, context):
        for i in range(5):
            yield statuses_updates_pb2.StatusUpdate(
                component_id=f"Component_{i}",
                status=statuses_updates_pb2.WARNING,
                message=f"Test warning #{i}",
                timestamp=int(time.time() * 1000)
            )
            time.sleep(1)

class CameraStreamingService(camera_stream_pb2_grpc.CameraStreamingServiceServicer):
    def GetPngCameraStream(self, request, context):
        for i in range(10):
            img = Image.new('RGB', (request.dimensions.width_px, request.dimensions.height_px), color=(0, 255, i*25 % 256))
            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                image_data = buf.getvalue()
            yield camera_stream_pb2.ImageFrame(
                image_data=image_data,
                timestamp=int(time.time() * 1000)
            )
            time.sleep(1)

class HMIConnection():
    def __init__(self):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))

        msg_to_hmi_pb2_grpc.add_FlightDataServiceServicer_to_server(FlightDataService(),  self.server)
        statuses_updates_pb2_grpc.add_DaaStatusServiceServicer_to_server(DaaStatusService(),  self.server)
        # camera_stream_pb2_grpc.add_CameraStreamingServiceServicer_to_server(CameraStreamingService(), server)

    def start_hmi_server(self):
        self.server.add_insecure_port('[::]:50051')
        self.server.start()
        print("HMI Server started on port 50051")
        # self.server.wait_for_termination() # this is a blocking call, it will keep the server running

    def stop_hmi_server(self):
        if self.server:
            self.server.stop(0)
            print("HMI Server stopped")

    def update_hmi_data(self, ownship, traffic, daidalus_output, system_status, controller_mode):
        # This method can be used to update flight data dynamically
        global ownship_data, traffic_data, daidalus_output_data, system_status_data, controller_mode_data

        with flight_data_lock:
            ownship_data = ownship
            traffic_data = traffic
            daidalus_output_data = daidalus_output
            system_status_data = system_status
            controller_mode_data = controller_mode
# HOW TO USE:

if __name__ == '__main__':
    # Create an instance of HMIConnection and start the server
    hmi_connection = HMIConnection()
    hmi_connection.start_hmi_server()

    # simulate 20 updates
    for i in range(20):

        ownship = {
            "id": "ownship",
            "is_active": True,
            "s": {"lat": 37.0, "lon": -122.0, "alt": 1000},
            "v": {"x": 10, "y": 0, "z": 0}
        }

        traffic = {
            "id": "1",
            "is_active": True,
            "s": {"lat": 37.001, "lon": -122.001, "alt": 950},
            "v": {"x": -5, "y": 2, "z": 0},
            "t_type": msg_to_hmi_pb2.ADSB,
            "bounding_box": {
                "image_data": b"",  # Placeholder for image data
                "timestamp": int(time.time() * 1000)
            }
        }

        daidalus = {
            "horizontal_guidance": [
                {
                    "lower_trk_deg": 45.0,
                    "upper_trk_deg": 135.0,
                    "region_type": 2  # REGION_TYPE_UNSPECIFIED = 0, REGION_TYPE_NONE = 1, REGION_TYPE_NEAR = 2;
                }
            ],
            "altitude_guidance": [
                {
                    "lower_alt_m": 1000.0,
                    "upper_alt_m": 1200.0,
                    "region_type": 2
                }
            ],
            "horizontal_speed_guidance": [
                {
                    "lower_h_spd_m_per_s": 50.0,
                    "upper_h_spd_m_per_s": 70.0,
                    "region_type": 2
                }
            ],
            "vertical_speed_guidance": [
                {
                    "lower_v_spd_m_per_s": -2.0,
                    "upper_v_spd_m_per_s": 2.0,
                    "region_type": 2
                }
            ],
            "intruders": [
                {
                    "id": "intruder_1",
                    "t2los_sec": 45.5,
                    "t2cpa_sec": 22.3,
                    "horizontal_separation_NM": 1.5,
                    "alert_level": 2
                },
                {
                    "id": "intruder_2",
                    # t2los_sec, t2cpa_sec, and horizontal_separation_NM omitted to show optionality
                    "alert_level": 1
                }
            ],
            "suggested_hdir_deg": 90.0,
            "suggested_hspd_knot": 120.0,
            "suggested_vspd_fpm": 500.0,
            "suggested_alt_ft": 5000.0
        }


        system_status = {
            "component_id": "System",
            "status": statuses_updates_pb2.OPERATIONAL,
            "message": "System is operational",
            "timestamp": int(time.time() * 1000)
        }

        controller_mode = {
            "mode" : 1
        }

        hmi_connection.update_hmi_data(ownship, [traffic], daidalus, system_status, controller_mode)

        time.sleep(1)  # Simulate some delay for updates

    # close connection
    hmi_connection.stop_hmi_server()