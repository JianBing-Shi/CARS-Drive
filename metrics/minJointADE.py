"""Backward-compatible Plan-R1 metric name with valid-scene counting."""
from metrics.displacement import JointDisplacement


class minJointADE(JointDisplacement):
    endpoint = False
