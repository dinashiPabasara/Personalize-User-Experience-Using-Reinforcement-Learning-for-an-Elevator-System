
# Personalize-User-Experience-Using-Reinforcement-Learning-for-an-Elevator-System

This project introduces an intelligent elevator reservation system that leverages face recognition and reinforcement learning to optimize elevator usage and enhance user experience, particularly in high-traffic environments like universities and office buildings.

Key Features

User Identification via Face Recognition

Users are recognized in real-time using camera input and matched with pre-registered images.

Technologies used: DeepFace, FaceNet, and OpenCV.

Profile pictures are stored and retrieved from AWS S3 during user registration.

Priority-Based Access with Reinforcement Learning

A trained Actor-Critic RL model dynamically assigns a priority score (1–3) to each user.

3 = Professors / Senior Lecturers

2 = Lecturers / Management

1 = Students

Priority level influences elevator access timing and reservation efficiency.

Mobile App Integration

Users can log in / register (with profile image upload).

After login, users can:

Reserve elevators ahead of arrival.

Select entry and destination floors.

Specify waiting time preference, urgency level, and group size.

Submit feedback post-ride for model optimization.

Real-Time Data Synchronization

Uses Firebase Realtime Database to store and retrieve:

Reservation data

User feedback

Recognition and priority assignment results

Tech Stack

Frontend / Mobile App: Flutter (or similar)

Backend: Python, Firebase Realtime DB

Face Recognition: DeepFace, FaceNet, OpenCV

Machine Learning: Actor-Critic Reinforcement Learning (TensorFlow / PyTorch)

Cloud Storage: AWS S3 for profile image handling

System Workflow Overview

User Registers/Login on the app → uploads clear face image

User Reserves Elevator: Select time, floors, urgency, wait time, and group size

Camera Module Recognizes User in real-time at the elevator

Reinforcement Learning Model assigns priority level

Elevator Access is Granted based on priority and reservation

User Feedback is collected for future optimization

https://github.com/user-attachments/assets/1c48c46a-9719-40d8-89d6-2989f9a4e51d



https://github.com/user-attachments/assets/e9a7ee2d-4106-4fe5-b2b9-b25c86769892


