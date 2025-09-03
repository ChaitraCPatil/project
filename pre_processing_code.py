!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="halOT0H6IuDCq08wcMdz")
project = rf.workspace("chaitra-cc6kt").project("my-first-project-upzwf")
version = project.version(4)
dataset = version.download("yolov8")
                