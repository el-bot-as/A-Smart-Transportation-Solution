# A Smart Transportation Solution: Edge-AI Passenger Analytics

An embedded, real-time computer vision pipeline deployed on a Raspberry Pi to automate passenger ingress/egress tracking and dynamically calculate vehicle occupancy. By utilizing spatial downsampling and decoupled tracking state logic, this edge-computing system achieves robust real-time performance on resource-constrained hardware.

## System Architecture & Data Flow

The project is structured as a decentralized Edge-AI pipeline, processing all video capture, feature extraction, and mathematical inference locally on the device without cloud dependency.

```

[image_capture.py] ──(Captures 640x480 JPEGs)──> [ /dataset/ ]
│
[model_training.py] ──(HOG Feature Extraction)
│
[facial_recognition.py] <──(Loads 128D Vectors)── [ encodings.pickle ]

```

1. **Biometric Enrollment (image_capture.py):** Captures reference frames at a normalized resolution of 640x480. Includes a software debounce mechanism (0.3s) to prevent frame spamming and minimize database bloat.
2. **Feature Extraction (model_training.py):** Uses Histogram of Oriented Gradients (HOG) to extract facial landmarks, transforming facial structures into localized 128-dimensional mathematical vectors serialized into a flat pickle file.
3. **Real-Time Inference (facial_recognition.py):** Loads reference vectors into volatile memory, matches real-time camera arrays using Euclidean distance metrics, and manages tracking state machines.

## Technical Optimizations

### 4x Spatial Downsampling

Running deep-learning facial embedding lookups on a standard Full HD (1920x1080) stream would cause significant CPU bottlenecking on an embedded system, dropping throughput to less than 1 FPS.

- The pipeline utilizes a spatial scaling factor (cv_scaler = 4), downsampling the processing array to 480x270 pixels before execution.
- Linear processing overhead is reduced drastically, allowing high-speed vector matching while maintaining high-fidelity (1080p) rendering for user-facing visual output.

### Decoupled State Tracking Logic

Instead of nesting exit countdown loops within the synchronous camera detection loop, tracking states are fully decoupled:

- **Ingress:** Recognizing a registered face instantly initializes its timestamp in a state dictionary and increments the counter.
- **Egress (Independent Clock Evaluation):** The system independent-scans the active tracking dictionary at the end of every single frame. If a person is missing from the frame continuously for a threshold period (COOLDOWN_SECONDS = 10), they are systematically dropped from memory and the occupancy counter decrements.

### Dynamic Anonymous Headcount Offset

To track unregistered passengers without the compute overhead of continuous centroid tracking (like DeepSORT), the pipeline computes a dynamic per-frame snapshot offset:

$$\text{Total Passengers} = \text{Count}(\text{Active Registered Tracked Entries}) + \text{Count}(\text{Anonymous Faces in Current Frame})$$

This mathematically prevents counter drifting or ghost-counting when unregistered individuals move out of camera frame angles.

## Tech Stack & Requirements

- **Hardware:** Raspberry Pi, PiCamera V3 / Compatible Modules
- **Runtime Environment:** Python 3.x
- **Core Libraries:** opencv-python, face_recognition (dlib-backed), picamera2, numpy, imutils

## Installation & Usage

1. **Clone the Repository:**

   ```
   git clone [https://github.com/el-bot-as/A-Smart-Transportation-Solution.git](https://github.com/el-bot-as/A-Smart-Transportation-Solution.git)
   cd A-Smart-Transportation-Solution

   ```

2. **Enroll Registered Passengers:**
   Run the capture script and press SPACE to collect face samples:

```
python image_capture.py



3. **Train the Model:**
Extract HOG features and compile your vector database:
```

python model_training.py

4. **Deploy the System:**
   Initialize the real-time smart transit system monitoring screen:

```
python facial_recognition.py
```
