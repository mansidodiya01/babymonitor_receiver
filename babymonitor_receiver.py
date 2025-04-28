import os
import serial
import base64
import time
import shutil
import cv2
import numpy as np

# === Config ===
serial_port = "/dev/cu.usbserial-130"  # Adjust to your system
baud_rate = 115200
base_dir = "received"
image_dir = os.path.join(base_dir, "images")
log_file = os.path.join(base_dir, "received_log.csv")

# === Clean Init ===
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)
os.makedirs(image_dir, exist_ok=True)

with open(log_file, "w") as f:
    f.write("Timestamp,Baby Detection,Cry Detection,Cropped Image\n")

# === Serial Setup ===
ser = serial.Serial(serial_port, baud_rate, timeout=2)
print("\U0001F4E1 Listening on Serial...")

# === State ===
current_image_name = None
expecting_chunks = 0
received_chunks = []
image_receiving = False
current_expected_image_name = None  # Holds image name from CSV

# === Enhancement Function (Reverse compression) ===
def reverse_compression_and_save(img_data, save_path):
    try:
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Image decoding failed")

        upscaled = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        smoothed = cv2.GaussianBlur(upscaled, (3, 3), 0)

        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        enhanced = cv2.filter2D(smoothed, -1, kernel)

        cv2.imwrite(save_path, enhanced)
        print(f"✅ Enhanced image saved: {save_path}")

    except Exception as e:
        print(f"❌ Enhancement failed: {e}")

# === Base64 Decode + Save ===
def save_image_from_base64(encoded_str, filename):
    try:
        padding_needed = (4 - len(encoded_str) % 4) % 4
        encoded_str += "=" * padding_needed
        img_data = base64.b64decode(encoded_str)
        img_path = os.path.join(image_dir, filename)
        reverse_compression_and_save(img_data, img_path)
    except Exception as e:
        print(f"❌ Failed to decode image {filename}: {e}")

# === Main Loop ===
try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        # === CSV Data ===
        if line.startswith("csv,"):
            print(f"🗒️ CSV Data: {line}")
            parts = line.strip().split(",")
            if len(parts) >= 5:
                current_expected_image_name = parts[4].strip().replace("?", "")
            with open(log_file, "a") as f:
                f.write(line.replace("csv,", "") + "\n")

        # === Heartbeat ===
        elif line.startswith("status:alive"):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"💓 Heartbeat received at {timestamp}")
            with open(log_file, "a") as f:
                f.write(f"{timestamp},Heartbeat,-,-\n")

        # === Image Start ===
        elif line.startswith("IMGSTART:"):
            try:
                expecting_chunks = int(line.split(":")[1])
                if current_expected_image_name:
                    current_image_name = current_expected_image_name
                else:
                    current_image_name = f"received_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                received_chunks = []
                image_receiving = True
                print(f"📥 Starting new image: {current_image_name} | Expected Chunks: {expecting_chunks}")
            except ValueError:
                print("⚠️ Invalid IMGSTART format")

        # === Image End ===
        elif line == "IMGEND":
            if image_receiving and len(received_chunks) == expecting_chunks:
                print(f"🧩 Reconstructing {current_image_name} from {len(received_chunks)} chunks")
                full_encoded = ''.join(received_chunks)
                save_image_from_base64(full_encoded, current_image_name)
            else:
                print(f"⚠️ Incomplete image: expected {expecting_chunks}, got {len(received_chunks)}")
            image_receiving = False
            received_chunks = []
            current_image_name = None
            current_expected_image_name = None

        # === CHUNK Data ===
        elif image_receiving:
            received_chunks.append(line.strip())
            print(f"📦 Received chunk {len(received_chunks)}/{expecting_chunks}")

        # === Unknown ===
        else:
            print(f"🔸 Unrecognized: {line}")

except KeyboardInterrupt:
    print("🛑 Stopped by user.")
finally:
    ser.close()