# 🔧 Troubleshooting Guide

## Quick Fixes

### 🎥 Camera Issues

**Problem**: Camera not detected or "Cannot open camera" error
```bash
Solutions:
✅ Close other apps using camera (Zoom, Teams, etc.)
✅ Check Windows Camera privacy settings
✅ Try different camera index: VideoCapture(1) instead of VideoCapture(0)
✅ Restart the application
```

**Problem**: Poor hand detection quality
```bash
Solutions:
✅ Improve lighting - face a window or add desk lamp
✅ Position hand 1-2 feet from camera
✅ Use solid background behind hand
✅ Clean camera lens
```

### 🖱️ Gesture Recognition Issues

**Problem**: Cursor not following finger
```bash
Solutions:
✅ Point with index finger only (close other fingers)
✅ Keep thumb down when pointing
✅ Ensure good contrast between hand and background
✅ Check if gesture shows "Mouse Control" status
```

**Problem**: Clicks not working
```bash
Solutions:
✅ Make deliberate pinch gesture (thumb touches index fingertip)
✅ Hold pinch for 1-2 seconds
✅ Check click cooldown hasn't triggered
✅ Verify "Clicking" status appears
```

**Problem**: Scrolling not responsive
```bash
Solutions:
✅ Keep wrist stationary while curling/extending fingers
✅ Make clear fist for scroll down
✅ Spread fingers wide for scroll up
✅ Hold gesture until "Scrolling" status appears
```

## Advanced Troubleshooting

### Performance Issues
```python
# Lower camera resolution in main.py
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)   # Instead of 1280
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Instead of 720
```

### Sensitivity Adjustments
```python
# Adjust in main.py
click_threshold = 60.0                   # Increase for less sensitive clicking
curl_threshold_high = 0.06              # Decrease for easier scroll down
curl_threshold_low = 0.18               # Increase for easier scroll up
```

### Multi-Camera Systems
```python
# Try different camera indices
cap = cv.VideoCapture(0)  # Built-in camera
cap = cv.VideoCapture(1)  # USB camera
cap = cv.VideoCapture(2)  # External camera
```

## System-Specific Issues

### Windows
- **Camera Permission**: Settings → Privacy → Camera → Allow apps to access camera
- **Multiple Cameras**: Device Manager → Cameras → Check which devices are active
- **Antivirus**: Some antivirus software blocks camera access

### macOS
- **Camera Permission**: System Preferences → Security & Privacy → Camera
- **Python Access**: Make sure terminal/IDE has camera permissions
- **Hardware**: Some MacBooks have camera hardware switches

### Linux
- **Permissions**: Add user to video group: `sudo usermod -a -G video $USER`
- **V4L Utils**: Install video4linux utilities: `sudo apt install v4l-utils`
- **List Cameras**: `v4l2-ctl --list-devices`

## Getting Help

### Enable Debug Mode
```python
# Add to main.py for more verbose output
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Rerun Visualization
```bash
# Use Rerun to debug gesture detection
python main.py
# Open Rerun viewer to see hand tracking data
```

### Common Error Messages

**"ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install opencv-python
```

**"ModuleNotFoundError: No module named 'mediapipe'"**
```bash
pip install mediapipe
```

**"ImportError: DLL load failed"**
```bash
# Windows: Install Visual C++ Redistributable
# Or try: pip install --upgrade opencv-python
```

## Still Having Issues?

1. **Check Requirements**: Ensure all dependencies are installed
2. **Test Camera**: Try with other camera apps first
3. **Update Drivers**: Update camera and graphics drivers
4. **System Restart**: Sometimes helps with camera access issues
5. **GitHub Issues**: Report bugs with system info and error messages

## Performance Optimization

### For Slower Systems
```python
# Reduce processing load
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,  # Lower confidence
    min_tracking_confidence=0.3    # Lower tracking
)
```

### For Better Accuracy
```python
# Increase detection quality
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,  # Higher confidence
    min_tracking_confidence=0.7    # Higher tracking
)
```
