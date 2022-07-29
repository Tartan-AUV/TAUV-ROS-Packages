from tauv_mission_manager.mission_utils import Mission, Task, TaskParams
from tauv_mission_manager.tasks import Dive, Square

class SquareMission(Mission):
    x = 3
    def __init__(self, params: TaskParams) -> None:
        self.p = params
        self.p.status("I'm being born :O")

        self.dive = Dive(params)
        self.square = Square(params)

    def run(self) -> None:
        self.p.status("hello")
        self.dive.run()
        self.square.run()

    def cancel(self) -> None:
        self.p.status("byeeee")
        self.dive.cancel()
        self.square.cancel()
