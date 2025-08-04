# AI-Powered Air Mouse with Computer Vision

A sophisticated computer vision project that turns your webcam into an air mouse system. Control your computer using hand gestures - point to move the cursor, pinch to click, and curl/extend fingers to scroll. Built with MediaPipe, OpenCV, and enhanced with Rerun SDK for real-time visualization and debugging.

## 🌟 Features

- 🖱️ **Air Mouse Control** - Move cursor with index finger pointing
- 🤏 **Gesture-Based Clicking** - Pinch thumb and index finger to click
- � **Finger-Curl Scrolling** - Curl fingers to scroll down, extend to scroll up
- 👁️ **Real-Time Hand Tracking** - MediaPipe-powered hand landmark detection
- � **Rerun Visualization** - Advanced debugging and analysis with 3D hand tracking
- 🎯 **High Precision** - 1280x720 camera resolution for accurate control
- 🛡️ **Smart Click Management** - Prevents accidental multiple clicks
- � **Gesture Stability** - Buffer system prevents jittery gesture detection
- � **Visual Feedback** - Real-time gesture recognition indicators
- ⚡ **Optimized Performance** - Efficient processing for smooth operation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- A webcam or camera device
- Windows, macOS, or Linux
- Basic understanding of computer vision concepts

### Installation

1. **Clone and enter the project directory:**
   ```powershell
   cd ai-agent
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Ensure your camera is connected and working**

### 🎯 Usage

#### 1. Launch Air Mouse System
```powershell
python main.py
```

#### 2. Hand Gestures
- **👉 Point with index finger** - Move mouse cursor
- **🤏 Pinch thumb + index** - Left click
- **✊ Make fist, curl fingers** - Scroll down (keep wrist still)
- **🖐️ Extend all fingers** - Scroll up (keep wrist still)
- **❌ Press 'q'** - Quit application

#### 3. Run Examples (if available)
```powershell
python examples.py
```

## 🔥 Computer Vision Features

### Hand Tracking System
```python
# Powered by MediaPipe Hands
- 21 hand landmarks per hand
- Real-time finger position detection
- Multi-hand support (optimized for single hand)
- Robust tracking in various lighting conditions
```

### Gesture Recognition
```python
# Smart gesture classification system
- Index finger pointing → Mouse control
- Thumb-index pinch → Click detection  
- Finger curl analysis → Scroll direction
- Wrist-stationary scrolling → Prevents drift
```

### Rerun SDK Integration
```python
# Advanced visualization and debugging
- 3D hand landmark visualization
- Real-time gesture state monitoring
- Scroll system analysis charts
- Click detection distance tracking
- System performance metrics
```

## 🛠️ System Components

### Core Files

- **`main.py`** - Main air mouse application with Rerun visualization
- **`clickmanager.py`** - Click cooldown management system
- **`examples.py`** - Demo scripts and usage examples
- **`pytorch_gpt4.py`** - Additional AI integration utilities
- **`requirements.txt`** - Python dependencies list

### Key Dependencies

- **OpenCV** - Camera input and image processing
- **MediaPipe** - Hand tracking and landmark detection  
- **PyAutoGUI** - System mouse and keyboard control
- **Rerun SDK** - Advanced visualization and debugging
- **NumPy** - Mathematical operations and array handling

## 📊 What You Can Do

### 🖱️ Complete Mouse Control
- Move cursor with natural pointing gestures
- Precise clicking with pinch detection
- Smooth scrolling with finger curl/extension
- Full screen coverage with edge expansion

### 📈 Advanced Debugging with Rerun
- Visualize hand landmarks in 3D space
- Monitor finger extension measurements
- Track scroll accumulator and thresholds
- Analyze gesture stability and performance

### 🎯 Gesture Recognition
- Real-time hand pose classification
- Multi-gesture support (point, click, scroll, fist)
- Stability buffers prevent jittery detection
- Visual feedback for all recognized gestures

### ⚡ Performance Optimization
- High-resolution camera input (1280x720)
- Efficient MediaPipe processing
- Smart coordinate mapping for screen coverage
- Optimized scroll rate and accumulation system

## 🌐 Supported Hardware

This air mouse system works with standard webcams and cameras:

- **Built-in Laptop Cameras** - Most modern laptops work great
- **USB Webcams** - Any OpenCV-compatible USB camera
- **External Cameras** - DSLR cameras with webcam mode
- **Network Cameras** - IP cameras with proper drivers
- **Phone Cameras** - Using apps like DroidCam or EpocCam

**Recommended Specs:**
- Resolution: 720p or higher for best accuracy
- Frame Rate: 30fps minimum for smooth tracking
- Good lighting conditions for reliable hand detection

## 🎓 Usage Tips

### 1. Optimal Setup
```bash
# Position camera at eye level
# Ensure good lighting (avoid backlighting)
# Keep hand 1-2 feet from camera
# Use consistent background when possible
```

### 2. Gesture Best Practices
```bash
# For mouse control: Point index finger clearly
# For clicking: Make deliberate pinch gesture
# For scrolling: Keep wrist stationary, curl/extend fingers
# For stability: Hold gestures for 2+ frames
```

### 3. Debugging with Rerun
```bash
# Launch with Rerun visualization
python main.py
# Monitor hand tracking quality in Rerun viewer
# Adjust thresholds based on finger extension data
```

## 🔧 Configuration

The system supports various configuration options by modifying the code:

```python
# Camera settings (in main.py)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)   # Camera width
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)   # Camera height
cap.set(cv.CAP_PROP_FPS, 30)             # Frame rate

# Gesture detection thresholds
click_threshold = 40.0                    # Pinch sensitivity
curl_threshold_high = 0.08               # Scroll down sensitivity
curl_threshold_low = 0.15                # Scroll up sensitivity

# System behavior
cooldown_seconds = 0.1                   # Click cooldown
gesture_buffer_threshold = 2             # Stability frames
scroll_rate = 0.4                        # Scroll speed
```

## 🚨 Troubleshooting

### Camera Issues
- ✅ Ensure camera is connected and not used by other applications
- ✅ Check camera permissions in Windows/macOS privacy settings
- ✅ Try different camera indices (0, 1, 2) if multiple cameras exist
- ✅ Verify camera works with other applications first

### Hand Detection Issues
- ✅ Improve lighting conditions - avoid shadows and backlighting
- ✅ Position hand 1-2 feet from camera for optimal detection
- ✅ Ensure good contrast between hand and background
- ✅ Clean camera lens if image appears blurry

### Gesture Recognition Problems
- ✅ Hold gestures steady for at least 2 frames
- ✅ Make deliberate, clear hand positions
- ✅ Adjust sensitivity thresholds in the code if needed
- ✅ Use Rerun visualization to debug finger extension values

### Performance Issues
- ✅ Close other resource-intensive applications
- ✅ Lower camera resolution if system is slow
- ✅ Ensure sufficient lighting to help MediaPipe processing
- ✅ Update graphics drivers for better performance

## 🎯 Next Steps

Ready to enhance your air mouse experience? Try these:

1. **Customization**: Adjust gesture thresholds for your hand size and preferences
2. **Multi-Hand**: Enable multi-hand tracking for advanced gestures
3. **New Gestures**: Add custom gestures like right-click or double-click
4. **Integration**: Combine with other applications for specialized control
5. **Hardware**: Experiment with different cameras for optimal tracking

## 📜 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Ideas for contributions:
- Additional gesture recognition (peace sign, thumbs up, etc.)
- Right-click and double-click gesture support
- Gesture customization interface
- Multi-monitor support improvements
- Mobile app companion for remote control
- Integration with accessibility tools

Feel free to submit issues and enhancement requests!
