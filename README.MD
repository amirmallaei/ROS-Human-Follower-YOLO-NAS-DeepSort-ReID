# Human Following Robot with YOLO-NAS, DeepSort, and Smart Re-Identification

This repository is the official implementation for the Master's thesis and paper, **"Enhancing Robot Vision: A Lightweight Statistical Method for Subject Re-Identification."**

This project demonstrates a complete, real-time system where a robot intelligently follows a person by integrating the power of **Yolo-NAS** for detection, **DeepSort** for tracking, and a custom statistical method for robustly re-identifying the target if they are lost.

---

## ⚙️ Core Technologies & The Integration Pipeline

The success of this project lies in its intelligent pipeline, where each component is used for its specific strength.

1.  **YOLO-NAS: The Spotter (Detection)**
    -   **Role:** For every camera frame, YOLO-NAS acts as the system's eyes, instantly identifying the location of every person in view. It's incredibly fast but has no memory—it only knows what it sees in the present moment.
    -   **Module:** `detection.py`

2.  **DeepSort: The Tracker (Short-Term Memory)**
    -   **Role:** DeepSort receives the raw detections from YOLO-NAS and intelligently assigns a consistent `track_id` to each person. It uses motion prediction and appearance to know that the person it called `ID: 5` in the last frame is the same person in the current frame, even if they've moved.
    -   **Module:** `tracking.py`

3.  **Our Re-ID Method: The Recognizer (Long-Term Memory)**
    -   **Role:** This is the system's "smart memory" that activates only when the tracker fails. If the target's `track_id` from DeepSort is lost (because they were occluded or left the frame), our lightweight Re-ID logic is triggered.
    -   **How it Works:** It compares a saved HSV color signature of the original target against the *other tracked individuals* from DeepSort to find a definitive match and resume following.
    -   **Modules:** `main_node.py` and `utils.py`

### Visual Workflow
```[Camera Frame]
      |
      V
[YOLO-NAS: Detects "a person is here"]
      |
      V
[DeepSort: Labels that person as "ID: 5"]
      |
      V
[Our Logic: Follows "ID: 5" or, if lost, uses Re-ID to find them again]
```

## ✨ Key Features

*   **State-of-the-Art Perception:** Integrates YOLO-NAS and DeepSort for highly accurate, real-time detection and tracking.
*   **Smooth & Responsive Following:** The robot uses a proportional controller for fluid, natural motion instead of jerky movements.
*   **Efficient Re-Identification:** The custom statistical Re-ID method runs at 25+ FPS, making the system ideal for robots with limited computational power.
*   **Highly Organized Code:** Thoughtfully structured into modules for detection, tracking, and control, making it easy to understand and extend.

## 🚀 Getting Started

Follow these steps to get the project up and running on your own ROS-enabled machine.

### Prerequisites

*   Ubuntu 20.04 (or 18.04)
*   ROS Noetic (or Melodic)
*   A working Catkin workspace (e.g., `~/catkin_ws`)
*   Python 3.8+

### 1. Place the Package
Copy this entire project folder into your Catkin workspace's `src` directory.
```bash
# Example:
cp -r /path/to/this_project_folder ~/catkin_ws/src/
```

### 2. Install Dependencies
All required Python libraries are listed in `requirements.txt`.
```bash
cd ~/catkin_ws/src/your_project_folder/
pip install -r requirements.txt
```

### 3. Make Scripts Executable
This is a crucial step! You must give ROS permission to run the Python scripts.
```bash
cd ~/catkin_ws/src/your_project_folder/scripts/
chmod +x *.py
chmod +x robot_camera_publisher/*.py
```

### 4. Build Your Workspace
Compile the project using `catkin_make`.
```bash
cd ~/catkin_ws/
catkin_make
source devel/setup.bash
```

## ▶️ How to Run

Execute each command in a **separate terminal**.

**Terminal 1: Start ROS Core**
```bash
roscore
```

**Terminal 2: Launch Your Camera Node**
Run your camera node, which must publish `sensor_msgs/CompressedImage` messages to the `/center_cam` topic.
```bash
# Example command, use your actual node
rosrun your_project_folder your_camera_node.py
```

**Terminal 3: Launch the Human Follower Node**
Start the main application.
```bash
rosrun your_project_folder main_node.py```
An OpenCV window will pop up, and your robot will spring to life!

## 🔧 Configuration & Tuning

Easily tune the robot's behavior by editing these parameters.

#### Re-Identification Strictness (`main_node.py`)
-   `self.bhattacharyya_threshold = 0.35`: The core of your Re-ID logic. A strict value to minimize false positives, as documented in our paper.

#### Robot "Personality" (`control.py`)
-   `P_ANGULAR_GAIN` & `P_LINEAR_GAIN`: Increase for a more aggressive robot; decrease for a more cautious one.
-   `TARGET_Y2`: Controls the following distance.
-   `MAX_LINEAR_SPEED` & `MAX_ANGULAR_SPEED`: Safety limits for your robot.



## 📬 Contact

Have questions, ideas, or want to collaborate? Feel free to reach out!

**Amir Mallaei** - [amirmallaei@gmail.com](mailto:amirmallaei@gmail.com)
```