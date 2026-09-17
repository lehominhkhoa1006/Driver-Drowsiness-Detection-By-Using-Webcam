# Driver Drowsiness Detection by Using Webcam

This project presents a webcam-based driver drowsiness detection system developed as a group project for the Special Project (EL) course at Ho Chi Minh City University of Technology and Education (HCMUTE) during the 2023–2024 academic year. The system uses computer vision and facial landmark analysis to monitor the driver's eye state in real time and provide an alert when prolonged eye closure is detected.

<p align="center">
  <img src="media/drowsiness_detection_preview.gif" alt="Driver Drowsiness Detection Demo" width="400"><br>
  <em>Figure 1. Real-time webcam-based eye monitoring and drowsiness warning during system operation.</em>
</p>

## Overview

The system monitors the driver's eyes in real time using a webcam, facial landmarks, and the Eye Aspect Ratio (EAR). The detected eye landmarks are used to calculate the EAR for both eyes, while a predefined threshold and consecutive-frame counter are used to identify prolonged eye closure.

When the system detects sustained eye closure, it activates a GPIO-connected buzzer and displays a warning on the video stream. The project was developed as a proof-of-concept prototype focusing on camera-based eye-state monitoring and real-time drowsiness warning.
