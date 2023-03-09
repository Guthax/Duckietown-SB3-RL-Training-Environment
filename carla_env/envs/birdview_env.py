import math
import os
import subprocess
import argparse
import carla
import time
from carla_env.wrappers import *


class CarlaBirdView:
    def __init__(self, host="127.0.0.1", port=2000, fps=15, start_carla=True):
        """
        A class for obtaining a bird's-eye view image of a running CARLA environment and painting the waypoints.

        Parameters:
            - host (str): IP address of the CARLA host
            - port (int): Port used to connect to CARLA
            - viewer_res (tuple[int, int]): Resolution of the spectator camera as a (width, height) tuple
            - fps (int): FPS of the client. If fps <= 0 then use unbounded FPS.
            - start_carla (bool): Whether to automatically start CARLA when True. Note that you need to set the environment
            variable ${CARLA_ROOT} to point to the CARLA root directory for this option to work.
        """

        # Start CARLA from CARLA_ROOT
        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            carla_path = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ['-quality_level=Low']
            launch_command += ['-benchmark']
            launch_command += ["-fps=%i" % fps]
            # launch_command += ['-RenderOffScreen']
            launch_command += ['-prefernvidia']
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL)
            print("Waiting for CARLA to initialize")

            # ./CarlaUE4.sh -quality_level=Low -benchmark -fps=15 -RenderOffScreen
            time.sleep(5)

        # Setup gym environment
        self.fps = fps

        self.done = False
        self.extra_info = []
        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 1 / self.fps
            settings.synchronous_mode = True
            self.world.apply_settings(settings)
            self.client.reload_world(False)  # reload map keeping the world settings

            self.spectator = self.world.get_spectator()
            self.spectator.set_transform(sensor_transforms["birdview"])
        except Exception as e:
            self.close()
            raise e

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self):
        for i, point in enumerate(self.world.map.get_spawn_points()):
            begin = point.location + carla.Location(z=1.25)
            angle = math.radians(point.rotation.yaw)
            sign_x = -np.sin(angle)
            sign_y = np.cos(angle)
            end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
            self.world.debug.draw_arrow(begin, end, arrow_size=1, life_time=0, thickness=0.5, color=carla.Color(255, 0, 0))
            self.world.debug.draw_string(point.location + carla.Location(x=sign_x * 3, y=sign_y * 3, z=2), str(i),
                                         color=carla.Color(0, 0, 255), life_time=2.0)

    def step(self):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")
        # Tick game
        self.world.tick()

    def is_done(self):
        return self.done


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--host", default="localhost", type=str, help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    argparser.add_argument("--fps", default=20, type=int, help="FPS. Delta time between samples is 1/FPS")
    args = argparser.parse_args()

    # Create vehicle and actors for data collecting
    env = CarlaBirdView(host=args.host, port=args.port, fps=args.fps, start_carla=True)

    # While there are more images to collect
    while not env.is_done():
        # Take action
        env.step()
        env.render()

    # Destroy carla actors
    env.close()
