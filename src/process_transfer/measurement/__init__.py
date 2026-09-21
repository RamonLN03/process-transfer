"""Sensors and the observations they produce: what can be measured on a plant.

This package does not know the plant. A sensor receives the noise-free values it has
to measure and an instrument specification, and returns readings; ``Observations`` holds
the readings, the known inputs and the metadata an engineer would have. No rate law,
no true parameter and no exact state lives here, and nothing here may import
``process_transfer.simulation`` (``tests/test_boundaries.py``). Models of later
milestones read ``Observations``; they never read the simulator.
"""
