import cv2

def test_camera(device_id):
    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        print(f"Cannot open /dev/video{device_id}")
        return False
    print(f"Device /dev/video{device_id} opened successfully")
    
    # 设置低分辨率以降低资源需求
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to capture frame from /dev/video{device_id}")
            break
        print(f"Frame captured from /dev/video{device_id}, shape: {frame.shape}")
        
        # 显示帧
        cv2.imshow(f"Video {device_id}", frame)
        
        # 等待按键（1ms），按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

# 测试所有设备
for i in range(6):
    print(f"\nTesting /dev/video{i}...")
    test_camera(i)