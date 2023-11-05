from abc import ABC, abstractmethod


class Tracker(ABC):
    @abstractmethod
    def add_region(self, region):
        """Add region"""

    @property
    @abstractmethod
    def last_bound(self):
        """Tracker version"""
        ...

    @property
    @abstractmethod
    def tracker_version(self):
        """Tracker version"""
        ...

    @property
    @abstractmethod
    def frames_since_target_seen(self):
        """frames_since_target_seen version"""
        ...

    @property
    @abstractmethod
    def blank_frames(self):
        """blank_frames version"""
        ...

    @abstractmethod
    def predicted_velocity(self):
        """predicted_velocity version"""
        ...

    @property
    @abstractmethod
    def tracking(self):
        """tracking"""
        ...
