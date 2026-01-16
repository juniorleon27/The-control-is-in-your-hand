"""
Diagnostic script to check if all dependencies are installed correctly
Run this before running the main gesture controller
"""

import sys
import subprocess

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    print(f"\nChecking {package_name}...")
    
    # Check if installed via pip
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Extract version
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':')[1].strip()
                    print(f"  ✓ Installed version: {version}")
                    break
        else:
            print(f"  ✗ Not installed via pip")
            return False
    except Exception as e:
        print(f"  ✗ Error checking installation: {e}")
        return False
    
    # Try importing
    try:
        if import_name == 'cv2':
            import cv2
            print(f"  ✓ Import successful (OpenCV {cv2.__version__})")
        elif import_name == 'mediapipe':
            import mediapipe as mp
            print(f"  ✓ Import successful (MediaPipe {mp.__version__})")
            # Check for solutions module
            if hasattr(mp, 'solutions'):
                print(f"  ✓ mp.solutions available")
                if hasattr(mp.solutions, 'hands'):
                    print(f"  ✓ mp.solutions.hands available")
                else:
                    print(f"  ✗ mp.solutions.hands NOT available")
                    return False
            else:
                print(f"  ✗ mp.solutions NOT available")
                return False
        elif import_name == 'pyautogui':
            import pyautogui
            print(f"  ✓ Import successful (PyAutoGUI {pyautogui.__version__})")
        elif import_name == 'numpy':
            import numpy as np
            print(f"  ✓ Import successful (NumPy {np.__version__})")
        else:
            __import__(import_name)
            print(f"  ✓ Import successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error during import: {e}")
        return False

def check_camera():
    """Check if camera is accessible"""
    print("\nChecking camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"  ✓ Camera accessible")
                print(f"  ✓ Frame size: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
                return True
            else:
                print(f"  ✗ Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print(f"  ✗ Cannot open camera")
            print(f"  Try: ls /dev/video* (Linux) or check Device Manager (Windows)")
            return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    print("=" * 60)
    print("Hand Gesture Controller - Diagnostic Check")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    results = {}
    
    # Check all required packages
    results['opencv'] = check_package('opencv-python', 'cv2')
    results['mediapipe'] = check_package('mediapipe', 'mediapipe')
    results['pyautogui'] = check_package('pyautogui', 'pyautogui')
    results['numpy'] = check_package('numpy', 'numpy')
    
    # Check camera
    results['camera'] = check_camera()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_good = True
    for component, status in results.items():
        status_str = "✓ OK" if status else "✗ FAILED"
        print(f"{component:15} {status_str}")
        if not status:
            all_good = False
    
    print("=" * 60)
    
    if all_good:
        print("\n✓ All checks passed! You can run gesture_controller.py")
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        print("\nCommon solutions:")
        print("\n1. Reinstall MediaPipe:")
        print("   pip uninstall mediapipe")
        print("   pip install mediapipe")
        print("\n2. Reinstall all packages:")
        print("   pip install -r requirements.txt --force-reinstall")
        print("\n3. Check Python version (3.8+ recommended):")
        print(f"   Current: {sys.version}")
        print("\n4. On Windows, you may need Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")

if __name__ == "__main__":
    main()