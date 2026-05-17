import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.track import RaceTrack
from src.plot import plot_track
t = RaceTrack(1)
plot_track(t, save=False)

