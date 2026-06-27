import os
import cv2
import time
import pickle
import faiss
import numpy as np
import threading

from insightface.app import FaceAnalysis
import supervision as sv

# =====================================================
# CONFIG
# =====================================================

KNOWN_FACES_DIR = "known_faces"
SIMILARITY_THRESHOLD = 0.45

FAISS_INDEX_FILE = "faces.index"
NAMES_FILE = "names.pkl"
EMBEDDINGS_FILE = "embeddings.npy"

# =====================================================
# GLOBALS
# =====================================================

known_names = []
known_embeddings = []
faiss_index = None

latest_frame = None
latest_faces = []

frame_lock = threading.Lock()
face_lock = threading.Lock()

running = True
track_cache = {}
fps = 0

# =====================================================
# LOAD INSIGHTFACE
# =====================================================

print("Loading InsightFace...")
app = FaceAnalysis(providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("InsightFace Loaded")

# =====================================================
# LOAD KNOWN FACES
# =====================================================

def load_known_faces():
    global known_names, known_embeddings, faiss_index

    if (
        os.path.exists(FAISS_INDEX_FILE)
        and os.path.exists(NAMES_FILE)
        and os.path.exists(EMBEDDINGS_FILE)
    ):
        print("Loading saved database...")
        faiss_index = faiss.read_index(FAISS_INDEX_FILE)
        known_embeddings = np.load(EMBEDDINGS_FILE)
        with open(NAMES_FILE, "rb") as f:
            known_names = pickle.load(f)
        print(f"Loaded {len(known_names)} faces.")
        return

    print("Creating database...")
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for file in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, file)
        image = cv2.imread(path)
        if image is None:
            continue

        faces = app.get(image)
        if len(faces) == 0:
            print(f"No face found in {file}")
            continue

        embedding = faces[0].embedding.astype(np.float32)
        known_embeddings.append(embedding)
        known_names.append(os.path.splitext(file)[0])

    if not known_embeddings:
        print("Database empty. Place images in 'known_faces' directory.")
        # Create a dummy index to avoid crashes
        faiss_index = faiss.IndexFlatIP(512)
        return

    known_embeddings = np.array(known_embeddings, dtype=np.float32)
    faiss.normalize_L2(known_embeddings)
    dimension = known_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(known_embeddings)

    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, known_embeddings)
    with open(NAMES_FILE, "wb") as f:
        pickle.dump(known_names, f)
    print(f"Saved {len(known_names)} faces.")

# =====================================================
# RECOGNITION
# =====================================================

def recognize_embedding(embedding):
    if faiss_index is None or faiss_index.ntotal == 0:
        return "Unknown"
        
    embedding = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(embedding)
    scores, indices = faiss_index.search(embedding, 1)

    score = scores[0][0]
    idx = indices[0][0]

    if score >= SIMILARITY_THRESHOLD:
        return known_names[idx]
    return "Unknown"

# =====================================================
# HELPER: BOX IOU MATCHING
# =====================================================

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# =====================================================
# CAMERA THREAD
# =====================================================

def camera_thread():
    global latest_frame, running
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        with frame_lock:
            latest_frame = frame.copy()
            
    cap.release()

# =====================================================
# RECOGNITION THREAD
# =====================================================

def recognition_thread():
    global latest_faces, running, track_cache
    tracker = sv.ByteTrack()

    while running:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        faces = app.get(frame)
        boxes = []
        scores = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            boxes.append([x1, y1, x2, y2])
            scores.append(face.det_score)

        if len(boxes) == 0:
            with face_lock:
                latest_faces = []
            time.sleep(0.03)
            continue

        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(scores),
            class_id=np.zeros(len(boxes), dtype=int)
        )

        tracked = tracker.update_with_detections(detections)
        display_faces = []

        for i in range(len(tracked.xyxy)):
            track_id = int(tracked.tracker_id[i])
            tx1, ty1, tx2, ty2 = map(int, tracked.xyxy[i])

            # Find matching original insightface detection via IoU
            best_iou = 0
            matched_embedding = None
            
            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                iou = get_iou([tx1, ty1, tx2, ty2], [fx1, fy1, fx2, fy2])
                if iou > best_iou:
                    best_iou = iou
                    matched_embedding = face.embedding

            # Fallback if track doesn't overlap well with a current raw detection
            if matched_embedding is None or best_iou < 0.3:
                if track_id in track_cache:
                    name = track_cache[track_id]["name"]
                    track_cache[track_id]["last_seen"] = time.time()
                else:
                    name = "Unknown"
            else:
                if track_id not in track_cache:
                    name = recognize_embedding(matched_embedding)
                    track_cache[track_id] = {
                        "name": name,
                        "last_seen": time.time()
                    }
                else:
                    track_cache[track_id]["last_seen"] = time.time()
                    name = track_cache[track_id]["name"]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            display_faces.append((tx1, ty1, tx2, ty2, name, track_id, color))

        # Cleanup old tracks
        now = time.time()
        track_cache = {tid: data for tid, data in track_cache.items() if now - data["last_seen"] <= 5}

        with face_lock:
            latest_faces = display_faces
            
        time.sleep(0.01) # Yield CPU control

# =====================================================
# STARTUP
# =====================================================

load_known_faces()

threading.Thread(target=camera_thread, daemon=True).start()
threading.Thread(target=recognition_thread, daemon=True).start()

# =====================================================
# DISPLAY LOOP
# =====================================================

frame_count = 0
fps_start = time.time()

while True:
    with frame_lock:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        frame = latest_frame.copy()

    with face_lock:
        faces = list(latest_faces) # explicit copy of elements

    for (x1, y1, x2, y2, name, track_id, color) in faces:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} ID:{track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    frame_count += 1
    elapsed = time.time() - fps_start
    if elapsed >= 1:
        fps = frame_count
        frame_count = 0
        fps_start = time.time()

    cv2.putText(
        frame,
        f"FPS: {fps}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    cv2.imshow("InsightFace + ByteTrack", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        running = False
        break

cv2.destroyAllWindows()