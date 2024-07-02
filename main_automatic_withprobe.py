import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Function to read the video and create frames
def read_video_and_create_frames(video_path, output_dir, max_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f'Extracted {frame_idx} frames from the video.')
    return frame_idx

# Function to manually select the center of the vessel
def select_center(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return None

    center = []

    def draw_center(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            center.append((x, y))
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Select Center", image)

    cv2.namedWindow("Select Center")
    cv2.setMouseCallback("Select Center", draw_center)
    cv2.imshow("Select Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if center:
        return center[0]
    else:
        return None

# Function to create a mask to exclude the probe region
def create_probe_mask(center, radius, shape):
    mask = np.ones(shape, dtype=np.uint8) * 255
    cv2.circle(mask, center, radius, 0, -1)
    return mask

# Function to find the inner wall contour based on the center and mask
def find_inner_wall_contour(frame_gray, center, mask, debug=False):
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    blurred = cv2.GaussianBlur(frame_gray, (7, 7), 0)
    enhanced = cv2.equalizeHist(blurred)
    
    # Adaptive thresholding
    # adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                        cv2.THRESH_BINARY_INV, 21, 3)
    _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
     # Morphological operations to remove noise and close gaps
    kernel = np.ones((5, 5), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    #morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=5)
    '''
    # Focus on the region of interest (closer to the center)
    height, width = morph.shape
    mask_inner = np.zeros_like(morph)
    cv2.circle(mask_inner, center, int(min(height, width) * 0.5), 255, -1)  # Inner circle for ROI
    morph = cv2.bitwise_and(morph, morph, mask=mask_inner) '''
    
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debugging: visualize intermediate steps
    if debug:
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Enhanced", enhanced)
        cv2.imshow("Adaptive Threshold", thresh)
        cv2.imshow("Morphological", morph)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ''' if contours:
        center_x, center_y = center
        # Filter contours based on area and circularity
        filtered_contours = [contour for contour in contours if 100 < cv2.contourArea(contour) < 5000]
        circular_contours = [contour for contour in filtered_contours if cv2.arcLength(contour, True) != 0 and cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2) > 0.05]
        if circular_contours:
            main_contour = max(circular_contours, key=cv2.contourArea)
        else:
            main_contour = max(filtered_contours, key=cv2.contourArea)
        return main_contour
    else:
        return None '''
    
    if contours:
        center_x, center_y = center
        distances = [cv2.pointPolygonTest(contour, (center_x, center_y), True) for contour in contours]
        max_index = np.argmax(distances)
        main_contour = contours[max_index]
        return main_contour
    else:
        return None

# Function to process a frame and draw the inner wall contour
def process_frame_with_contour(frame_path, center, mask, output_path, debug=False):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Could not read frame {frame_path}")
        return
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inner_contour = find_inner_wall_contour(frame_gray, center, mask, debug)

    if inner_contour is not None:
        cv2.drawContours(frame, [inner_contour], -1, (0, 0, 255), 2)  # Draw contour in red

    combined_image = np.hstack((cv2.imread(frame_path), frame))
    cv2.imwrite(output_path, combined_image)

# Main function
def main():
    video_path = '/home/bazzi/TEVG/FSG/Inverse_model/230206_Martini_IVUS_Low_IVC_with_breathing.avi'  # Update with your video path
    frames_dir = '/home/bazzi/TEVG/FSG/Inverse_model/frames'
    processed_dir = '/home/bazzi/TEVG/FSG/Inverse_model/processed_frames'
    max_frames = 10

    # Step 1: Read the video and create frames
    num_frames = read_video_and_create_frames(video_path, frames_dir, max_frames)
    if num_frames == 0:
        return

    # Step 2: Manually select the center of the vessel
    first_frame_path = os.path.join(frames_dir, 'frame_0000.png')
    center = select_center(first_frame_path)
    if center is None:
        print("Error: Center not selected.")
        return

    probe_radius = 27  # Adjust this radius as needed

    # Step 3: Create a mask to exclude the probe region
    first_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
    if first_frame is None:
        print(f"Error: Could not read first frame {first_frame_path}")
        return
    mask = create_probe_mask(center, probe_radius, first_frame.shape)
    cv2.imwrite('mask.png', mask)  # Save the mask for visualization

    # Step 4: Process frames to find and highlight the inner wall contour
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for frame_file in sorted(os.listdir(frames_dir)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(processed_dir, frame_file)
            process_frame_with_contour(frame_path, center, mask, output_path, debug=True)

    # Display the first processed frame
    processed_frame_path = os.path.join(processed_dir, 'frame_0000.png')
    plt.imshow(cv2.cvtColor(cv2.imread(processed_frame_path), cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()