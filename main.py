#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile


# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Initialize devices
ev3 = EV3Brick()
left_motor = Motor(Port.B)
right_motor = Motor(Port.C)
ultrasonic = UltrasonicSensor(Port.S1)

# Settings
NUM_STEPS = 40        
DEGREES_PER_STEP = 18        
ROTATION_SPEED = 100         # deg/sec per motor
WAIT_BETWEEN = 1000          # ms between scans

signature = []

# Open file for writing
with open("/home/robot/Mindstorm_localization/test_signature.csv", "w") as file:
    file.write("Angle,Distance_mm\n")

    for step in range(NUM_STEPS):
        angle = step * 9

        distance = ultrasonic.distance()
        signature.append(distance)
        file.write("{},{}\n".format(angle, distance))
        ev3.screen.clear()
        ev3.screen.print("{}°: {}mm".format(angle, distance))

        # Pivot around right wheel only
        left_motor.run_angle(ROTATION_SPEED, DEGREES_PER_STEP * 2, then=Stop.BRAKE, wait=True)

        wait(WAIT_BETWEEN)

# Done
left_motor.stop()
right_motor.stop()
ev3.speaker.beep()
ev3.screen.print("Scan complete!")
