import rospy
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from threading import Event
from motion_client import MotionClient
from actuator_client import ActuatorClient
from map_client import MapClient
from transform_client import TransformClient
import time


@dataclass
class TaskResources:
    motion: MotionClient
    actuators: ActuatorClient
    map: MapClient
    transforms: TransformClient


TaskStatus = IntEnum


@dataclass
class TaskResult:
    status: TaskStatus


class Task(ABC):

    def __init__(self, timeout = None) -> None:
        self._cancel_event: Event = Event()
        self._cancel_complete_event: Event = Event()
        self._timeout = timeout

    @abstractmethod
    def run(self, resources: TaskResources) -> TaskResult:
        pass

    @abstractmethod
    def _handle_cancel(self, resources: TaskResources):
        pass

    def cancel(self):
        rospy.logdebug('cancel setting cancel_event')
        self._cancel_event.set()
        self._cancel_complete_event.wait()

    def _check_cancel(self, resources: TaskResources):
        rospy.logdebug('_check_cancel checking cancel_event')
        if self._cancel_event.is_set():
            rospy.logdebug('_check_cancel cancel_event is set, running _handle_cancel')
            self._cancel_event.clear()

            self._handle_cancel(resources)

            self._cancel_complete_event.set()
        
    def _set_timeout(self):
        if(not self._timeout is None):
            self._end_time = time.time()+self._timeout
        else:
            self._end_time = None

    def _check_timeout(self):
        if(not self._end_time is None):
            return (time.time()<self._end_time)
        return False