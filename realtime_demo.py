import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import json
import os

# ==========================================
# 1. CONSTANTS & INDICES (Aligned with Kaggle)
# ==========================================
LIPS_IDXS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LEFT_HAND_IDXS = np.arange(468, 489)
RIGHT_HAND_IDXS = np.arange(522, 543)
LEFT_POSE_IDXS = [500, 502, 504, 506, 508]
RIGHT_POSE_IDXS = [501, 503, 505, 507, 509]

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.20
STABILITY_THRESHOLD = 5  # Frames

# ==========================================
# 2. CUSTOM MODEL LAYERS
# ==========================================
class LandmarkEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
    def build(self, input_shape):
        flat_dim = input_shape[2] * input_shape[3]
        self.dense1 = tf.keras.layers.Dense(self.units, activation='gelu')
        self.dense2 = tf.keras.layers.Dense(self.units)
        self.dense1.build((None, input_shape[1], flat_dim))
        self.dense2.build((None, input_shape[1], self.units))
        super().build(input_shape)
    def call(self, x):
        P, C = x.shape[2], x.shape[3]
        x_flat = tf.reshape(x, (-1, x.shape[1], P * C))
        dense_out = self.dense2(self.dense1(x_flat))
        is_empty = tf.reduce_sum(tf.abs(x_flat), axis=-1, keepdims=True) == 0.0
        return tf.where(is_empty, tf.zeros_like(dense_out), dense_out)

class SoftmaxWeightedFusion(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
    def build(self, input_shape):
        self.landmark_weights = self.add_weight(shape=(len(input_shape),), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, inputs):
        x = tf.stack(inputs, axis=-1)
        weights = tf.nn.softmax(self.landmark_weights)
        return tf.reduce_sum(x * weights, axis=-1)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, units, sequence_length, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.sequence_length = sequence_length
    def build(self, input_shape):
        self.pos_emb = self.add_weight(shape=(self.sequence_length, self.units), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        return x + self.pos_emb

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)
    def call(self, inputs):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        return inputs * nn[:, None, :]

# GetItem layer is created dynamically by Keras during slicing, must be registered explicitly
try:
    from keras.src.ops.numpy import GetItem
except ImportError:
    try:
        from tensorflow.keras.src.ops.numpy import GetItem
    except ImportError:
        GetItem = None

CUSTOM_OBJECTS = {
    'LandmarkEmbedding': LandmarkEmbedding,
    'SoftmaxWeightedFusion': SoftmaxWeightedFusion,
    'PositionalEmbedding': PositionalEmbedding,
    'ECA': ECA
}

if GetItem is not None:
    CUSTOM_OBJECTS['GetItem'] = GetItem

# ==========================================
# 3. EXTRACTION & PREPROCESSING LOGIC
# ==========================================
def resize_sequence(frames_arr, target_length=30):
    """Resamples variable length frames to a fixed target length of 30 frames."""
    total_frames = len(frames_arr)
    if total_frames == 0:
        return np.zeros((target_length, 543, 3), dtype=np.float32)
    indices = np.linspace(0, total_frames - 1, target_length).astype(int)
    return np.array(frames_arr)[indices]
def extract_keypoints(results):
    """Extracts MediaPipe landmarks into Kaggle's (543, 3) format."""
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark[:468]]) if results.face_landmarks else np.full((468, 3), np.nan)
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark[:21]]) if results.left_hand_landmarks else np.full((21, 3), np.nan)
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark[:21]]) if results.right_hand_landmarks else np.full((21, 3), np.nan)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark[:33]]) if results.pose_landmarks else np.full((33, 3), np.nan)
    
    # Pad face/pose/hands if needed to enforce correct shapes
    if face.shape[0] < 468:
        face = np.pad(face, ((0, 468 - face.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
    if left_hand.shape[0] < 21:
        left_hand = np.pad(left_hand, ((0, 21 - left_hand.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
    if right_hand.shape[0] < 21:
        right_hand = np.pad(right_hand, ((0, 21 - right_hand.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
    if pose.shape[0] < 33:
        pose = np.pad(pose, ((0, 33 - pose.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        
    return np.concatenate([face, left_hand, pose, right_hand], axis=0) # Shape: (543, 3)

def preprocess_live_sequence(sequence):
    """Applies the Kaggle Phase 2 Preprocessing to a (30, 543, 3) window."""
    data = np.array(sequence) # Shape: (30, 543, 3)
    
    lh_exists = np.sum(~np.isnan(data[:, LEFT_HAND_IDXS, 0]))
    rh_exists = np.sum(~np.isnan(data[:, RIGHT_HAND_IDXS, 0]))
    left_dominant = lh_exists >= rh_exists
    
    lips = np.copy(data[:, LIPS_IDXS, :])
    if left_dominant:
        hand_1 = np.copy(data[:, LEFT_HAND_IDXS, :])
        pose = np.copy(data[:, LEFT_POSE_IDXS + RIGHT_POSE_IDXS, :])
    else: 
        hand_1 = np.copy(data[:, RIGHT_HAND_IDXS, :]) * [-1, 1, 1]
        pose = np.copy(data[:, RIGHT_POSE_IDXS + LEFT_POSE_IDXS, :]) * [-1, 1, 1]
        lips = lips * [-1, 1, 1]
        
    # Local & Global Centering
    hand_1 = hand_1 - hand_1[:, 0:1, :]
    face_ref = lips[:, 0:1, :]
    lips = lips - face_ref
    pose = pose - face_ref
    
    clean_data = np.concatenate([lips, hand_1, pose], axis=1)
    clean_data = np.nan_to_num(clean_data, nan=0.0)
    
    # Velocity
    diff_data = np.diff(clean_data, axis=0, prepend=clean_data[0:1, :, :])
    features = np.concatenate([clean_data, diff_data], axis=-1)
    
    # Return as batch (1, 30, 71, 6)
    return np.expand_dims(features.astype(np.float32), axis=0)

# ==========================================
# 4. MAIN INFERENCE LOOP
# ==========================================
def main():
    print("Loading Vocabulary...")
    vocab_path = "sign_to_prediction_index_map.json"
    gloss_map = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r') as f:
            idx_map = json.load(f)
            gloss_map = {int(v): k for k, v in idx_map.items()}
    else:
        print(f"WARNING: {vocab_path} not found. Predictions will be raw numbers.")

    print("Loading Model...")
    model_path = "advanced_kaggle_transformer_250.h5" #"eca_conv1d_transformer_250.h5"
    if not os.path.exists(model_path) and os.path.exists(os.path.join("models", model_path)):
        model_path = os.path.join("models", model_path)
    model = None
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
        print(f"Model loaded successfully from {model_path}!")
    else:
        print(f"WARNING: {model_path} not found. Running in DRY RUN mode (no predictions).")

    # Tracking variables
    gesture_buffer = []
    is_recording = False
    cooldown_counter = 0
    COOLDOWN_LIMIT = 15  # Number of empty frames to wait before finalizing a sign (bridging dropouts)
    MIN_SIGN_FRAMES = 12  # Minimum frames to classify it as a sign (filters accidental movement)
    MAX_SIGN_FRAMES = 90  # Maximum frames before auto-triggering prediction (~3 seconds)
    
    current_prediction = "READY - Make a sign"
    current_confidence = 0.0
    top_k_predictions = []

    # Initialize MediaPipe and Webcam (reduced detection confidence to 0.3 for better overlap tracking)
    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        print("Starting webcam... Press 'q' to quit.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Flip frame horizontally to correct mirror effect (essential for coordinate mirroring alignment)
            frame = cv2.flip(frame, 1)

            # MediaPipe processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract keypoints
            keypoints = extract_keypoints(results)

            # Count detected hand landmarks for diagnostics
            lh_detected = np.sum(~np.isnan(keypoints[LEFT_HAND_IDXS, 0]))
            rh_detected = np.sum(~np.isnan(keypoints[RIGHT_HAND_IDXS, 0]))
            hand_present = (lh_detected > 0 or rh_detected > 0)

            # --- Gesture Accumulation Logic ---
            if hand_present:
                if not is_recording:
                    # Start of a new gesture
                    print("\n[START] Gesture detected. Recording...")
                    is_recording = True
                    cooldown_counter = 0
                    gesture_buffer = []
                    top_k_predictions = []
                
                gesture_buffer.append(keypoints)
                cooldown_counter = 0
                
                # Check for auto-trigger limit
                if len(gesture_buffer) >= MAX_SIGN_FRAMES:
                    print(f"[AUTO-TRIGGER] Gesture limit reached ({MAX_SIGN_FRAMES} frames). Predicting...")
                    is_recording = False
            
            elif is_recording:
                # Hand disappeared momentarily, accumulate and count cooldown
                gesture_buffer.append(keypoints)
                cooldown_counter += 1
                
                # Dynamic cooldown: use 15 frames if hands are raised near the face (occlusion bridging)
                # but cut off immediately after 3 frames if hands are lowered (snappy completion)
                current_cooldown_limit = 15
                if results.pose_landmarks:
                    try:
                        left_shoulder_y = results.pose_landmarks.landmark[11].y
                        right_shoulder_y = results.pose_landmarks.landmark[12].y
                        left_wrist = results.pose_landmarks.landmark[15]
                        right_wrist = results.pose_landmarks.landmark[16]
                        
                        left_raised = (left_wrist.y < left_shoulder_y + 0.15) if left_wrist.visibility > 0.5 else False
                        right_raised = (right_wrist.y < right_shoulder_y + 0.15) if right_wrist.visibility > 0.5 else False
                        
                        if not (left_raised or right_raised):
                            current_cooldown_limit = 3
                    except Exception:
                        current_cooldown_limit = 5
                else:
                    current_cooldown_limit = 5

                if cooldown_counter >= current_cooldown_limit:
                    # Gesture ended, slice off trailing empty frames
                    cleaned_buffer = gesture_buffer[:-cooldown_counter] if len(gesture_buffer) > cooldown_counter else gesture_buffer
                    
                    if len(cleaned_buffer) >= MIN_SIGN_FRAMES:
                        print(f"[PROCESS] Gesture ended. Total frames: {len(cleaned_buffer)}. Resampling to 30...")
                        
                        # Temporal scaling to exactly 30 frames
                        resized_seq = resize_sequence(cleaned_buffer, target_length=SEQUENCE_LENGTH)
                        processed_input = preprocess_live_sequence(resized_seq)
                        
                        if model:
                            res = model.predict(processed_input, verbose=0)[0]
                            
                            # Get top 5 predictions
                            top_indices = np.argsort(res)[-5:][::-1]
                            top_k_predictions = []
                            for idx in top_indices:
                                g = gloss_map.get(idx, f"Class {idx}")
                                c = res[idx]
                                top_k_predictions.append((g, c))
                            
                            best_gloss, best_conf = top_k_predictions[0]
                            print(f"[PREDICTION] Translated: {best_gloss:<15} | Confidence: {best_conf*100:5.1f}% | Raw Frames: {len(cleaned_buffer)}")
                            print("Top 5 predictions:")
                            for g, c in top_k_predictions:
                                print(f"  - {g:<15}: {c*100:5.1f}%")
                            
                            if best_conf > CONFIDENCE_THRESHOLD:
                                current_prediction = best_gloss
                                current_confidence = best_conf
                            else:
                                current_prediction = "Sign not recognized (Low Confidence)"
                                current_confidence = 0.0
                        else:
                            current_prediction = f"DRY RUN: Resampled {len(cleaned_buffer)} -> 30 frames"
                    else:
                        print(f"[DISCARD] Gesture too short ({len(cleaned_buffer)} frames).")
                    
                    # Reset states
                    gesture_buffer = []
                    is_recording = False
                    cooldown_counter = 0

            # --- Visual Overlay Status ---
            if is_recording:
                display_text = f"RECORDING: {len(gesture_buffer)} frames..."
                display_color = (0, 0, 255) # Red box for recording
            elif current_confidence > 0:
                display_text = f"{current_prediction} ({current_confidence*100:.1f}%)"
                display_color = (0, 255, 0) # Green box for translation
            else:
                display_text = current_prediction
                display_color = (245, 117, 16) # Orange box for ready state

            # --- Visual Overlay ---
            # Background Box
            cv2.rectangle(image, (0,0), (640, 50), display_color, -1)
            
            # Text
            cv2.putText(image, display_text, (10, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            # Draw Hand Detection Status Indicators
            lh_status = "LH: OK" if lh_detected > 0 else "LH: --"
            rh_status = "RH: OK" if rh_detected > 0 else "RH: --"
            cv2.putText(image, lh_status, (480, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, rh_status, (560, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw Top 5 List Overlay
            if not is_recording and len(top_k_predictions) > 0:
                # Draw a semi-transparent panel for Top 5
                overlay = image.copy()
                cv2.rectangle(overlay, (10, 60), (220, 195), (50, 50, 50), -1)
                cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
                
                cv2.putText(image, "Top 5 Likely:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                for i, (g, c) in enumerate(top_k_predictions):
                    pred_str = f"{i+1}. {g} ({c*100:.1f}%)"
                    color = (0, 255, 0) if i == 0 and c > CONFIDENCE_THRESHOLD else (200, 200, 200)
                    cv2.putText(image, pred_str, (20, 100 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

            # Draw some basic hand landmarks for visual feedback
            mp.solutions.drawing_utils.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp.solutions.drawing_utils.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow('Real-Time Sign Language Translation', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
