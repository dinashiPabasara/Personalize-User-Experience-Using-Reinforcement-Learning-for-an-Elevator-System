# --------------------------- IMPORTS ---------------------------
import boto3
import cv2
from deepface import DeepFace
import os
import threading
import time
import firebase_admin
from firebase_admin import credentials, db
import torch
import numpy as np
from queue import Queue
import pyttsx3
from datetime import datetime
from model_file import load_actor_model, load_scaler, predict_priority

# --------------------------- S3 SETUP ---------------------------
s3_client = boto3.client('s3')
bucket_name = 'elevatr-personalize'
profile_folder = 'profile_pictures/'
os.makedirs(profile_folder, exist_ok=True)
processed_files = set()

# --------------------------- FIREBASE ---------------------------
cred = credentials.Certificate('firebase_key')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://elevator-personalization-default-rtdb.firebaseio.com/'
})

# --------------------------- MODEL ---------------------------
state_size = 1
action_size = 4
model = load_actor_model('actor_designation_only.pth', state_size, action_size)
scaler = load_scaler('scaler_designation_only.pkl')
if torch.cuda.is_available():
    model = model.to('cuda')

# --------------------------- UTILS ---------------------------
def get_firebase_uid(user_id):
    try:
        all_users = db.reference('/users').get()
        for uid, user in (all_users or {}).items():
            if user.get("userId") == user_id:
                return user.get("firebaseUID")
    except Exception as e:
        print(f"[ERROR] get_firebase_uid: {e}")
    return None

def get_user_data(user_id):
    try:
        all_users = db.reference('/users').get()
        for uid, user in (all_users or {}).items():
            if user.get("userId") == user_id:
                return user
    except Exception as e:
        print(f"[ERROR] get_user_data: {e}")
    return None

def get_reservation(firebase_uid):
    try:
        all_res = db.reference('/reservations').get() or {}
        for uid, res_entries in all_res.items():
            if uid == firebase_uid:
                for res_id, res in res_entries.items():
                    return res_id, res
        return None, None
    except Exception as e:
        print(f"Firebase reservation fetch error: {e}")
        return None, None

# --------------------------- S3 Downloader ---------------------------
def download_profile_pics():
    while True:
        try:
            objs = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=profile_folder)
            for obj in objs.get('Contents', []):
                key = obj['Key']
                if not key.endswith('/'):
                    fn = os.path.basename(key)
                    if fn not in processed_files:
                        local = os.path.join(profile_folder, fn)
                        s3_client.download_file(bucket_name, key, local)
                        processed_files.add(fn)
                        print(f"Downloaded new profile: {fn}")
        except Exception as e:
            print(f"S3 download error: {e}")
        time.sleep(10)

# --------------------------- STATE ---------------------------
frame_queue = Queue(maxsize=2)
display_lines = []
recognized_data = None
recognized_successfully = False
lock = threading.Lock()
stop_event = threading.Event()

# --------------------------- CAMERA LOOP ---------------------------
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"[ERROR] Voice thread failed: {e}")

def capture_loop():
    global recognized_successfully, recognized_data
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Camera started.")

    recognition_displayed = False
    display_start_time = None

    while not stop_event.is_set():
        ret, frame = cam.read()
        if not ret:
            continue

        with lock:
            if recognized_successfully and display_lines:
                if not recognition_displayed:
                    print("[INFO] Displaying recognized user on camera window.")
                    display_start_time = time.time()
                    recognition_displayed = True

                    #  voice message with priority + destination
                    try:
                        user_id = recognized_data['userID']
                        priority = recognized_data.get('predictedPriority', 'unknown')
                        dest_floor = recognized_data.get('reservation', {}).get('destinationFloor', 'unknown')
                        
                        voice_msg = f"User {user_id} recognized. Priority score is {priority}. Going to floor {dest_floor}."
                        print(f"[VOICE] {voice_msg}")
                        
                        threading.Thread(target=speak_text, args=(voice_msg,), daemon=True).start()
                    except Exception as e:
                        print(f"[ERROR] Voice alert thread start failed: {e}")

                # Draw text lines 
                overlay_text_color = (0, 255, 0) 
                y0, dy = 35, 25
                for i, line in enumerate(display_lines):
                    y = y0 + i * dy
                    cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, overlay_text_color, 2)

        cv2.imshow('Elevator Camera', frame)

        if recognition_displayed and time.time() - display_start_time >= 10:
            print("[INFO] Closing camera window after 10 seconds display.")
            stop_event.set()
            break

        if not frame_queue.full():
            frame_queue.put(frame.copy())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cam.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

    # Upload recognized data to Firebase with timestamp
    if recognized_successfully and recognized_data:
        try:
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_path = f"/recognized_users_log/{recognized_data['userID']}/{timestamp_str}"
            db.reference(log_path).set(recognized_data)
            print(f"[UPLOAD] Data uploaded to Firebase: {log_path}")
        except Exception as e:
            print(f"[ERROR] Firebase upload failed: {e}")
    else:
        print("No recognized data found for Firebase upload.")


# --------------------------- INFERENCE LOOP ---------------------------
def inference_loop():
    global display_lines, recognized_data, recognized_successfully
    recognized = set()
    last_time = {}
    cooldown = 10
    threshold = 0.35

    print("[INFO] Inference loop started.")
    print("Available profiles:", os.listdir(profile_folder))

    while not stop_event.is_set():
        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        try:
            results = DeepFace.find(
                frame,
                db_path=profile_folder,
                enforce_detection=False,
                model_name='Facenet',
                detector_backend='opencv',
                distance_metric='cosine'
            )
        except Exception as e:
            print(f"[ERROR] DeepFace error: {e}")
            continue

        if not results or len(results[0]) == 0:
            print("[DEBUG] No match found.")
            continue

        match = results[0].iloc[0]
        matched_path = match['identity']
        distance = match['distance']

        if distance > threshold:
            print(f"[DEBUG] Closest match too far (distance: {distance:.4f}) > threshold ({threshold})")
            continue

        user_id = os.path.splitext(os.path.basename(matched_path))[0]
        now = time.time()

        if user_id in recognized and now - last_time.get(user_id, 0) < cooldown:
            print(f"[DEBUG] Skipping repeat recognition for {user_id}")
            continue

        print(f"[STEP] Match confirmed for: {user_id}")

        firebase_uid = get_firebase_uid(user_id)
        if not firebase_uid:
            print(f"[ERROR] No firebaseUID for user ID: {user_id}")
            continue

        res_id, reservation = get_reservation(firebase_uid)
        if not reservation:
            print(f"[ERROR] No reservation found at /reservations/{firebase_uid}")
            continue

        user_data = get_user_data(user_id)
        if not user_data:
            print(f"[ERROR] No user data found for userId: {user_id}")
            continue

        designation = user_data.get('designation', 'Unknown')
        name = user_data.get('name') or user_data.get('userName') or user_data.get('fullName') or 'Unknown'
        priority_score = predict_priority(designation, model, scaler)

        lines = [
            f"User Name: {name}",
            f"User ID: {user_id}",
            f"Designation: {designation}",
            f"Entry Floor: {reservation['entryFloor']}, Dest Floor: {reservation['destinationFloor']}",
            f"People: {reservation['numberOfPeople']}, Urgency: {reservation['urgencyLevel']}",
            f"Wait Pref: {reservation['waitingTimePreference']} min",
            f"Predicted Priority: {priority_score}"
        ]

        with lock:
            display_lines = lines
            recognized_data = {
                'userID': user_id,
                'userName': name,
                'firebaseUID': firebase_uid,
                'designation': designation,
                'reservation': reservation,
                'predictedPriority': priority_score
            }
            recognized_successfully = True

        recognized.add(user_id)
        last_time[user_id] = now
        print(f"[SUCCESS] All data ready for {user_id} ({name})")
        break

# --------------------------- MAIN ---------------------------
if __name__ == '__main__':
    threading.Thread(target=download_profile_pics, daemon=True).start()
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping application.")










