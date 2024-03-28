def video_to_frames(video_path):
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print("Error opening video file.")
        return []

    frames_folder = os.path.join(os.path.dirname(video_path), "frames")
    os.makedirs(frames_folder, exist_ok=True)

    success, image = vidcap.read()
    count = 0
    frame_paths = []

    while success:
        frame_path = os.path.join(frames_folder, f"{count}.png")
        cv2.imwrite(frame_path, image)
        frame_paths.append(frame_path)
        count += 1
        success, image = vidcap.read()

    vidcap.release()
    print(f"Frames saved to: {frames_folder}")
    
    return frame_paths