import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Function to read the video and create frames
def read_video_and_create_frames(video_path, output_dir, max_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f'Extracted {frame_idx} frames from the video.')

# Function to manually select the center of the vessel
def select_center(image_path):
    image = cv2.imread(image_path)
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

# Function to process a frame and remove the probe
def remove_probe(frame_path, center, mask, output_path):
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked_frame = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    
    # Create a black circle to cover the probe
    frame[mask == 0] = (0, 0, 0)
    
    cv2.imwrite(output_path, frame)

# Function to find the inner wall contour based on the center and mask
def find_inner_wall_contour(frame_gray, center, mask):
    frame_gray = cv2.bitwise_and(frame_gray, frame_gray, mask=mask)
    blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
    enhanced = cv2.equalizeHist(blurred)
    _, thresh = cv2.threshold(enhanced, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        center_x, center_y = center
        distances = [cv2.pointPolygonTest(contour, (center_x, center_y), True) for contour in contours]
        max_index = np.argmax(distances)
        main_contour = contours[max_index]
        return main_contour
    else:
        return None

# Function to smooth the contour using spline interpolation
def smooth_contour(contour, num_points=100):
    contour = contour[:, 0, :]  # Convert to Nx2 format
    tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True)
    u_new = np.linspace(u.min(), u.max(), num_points)
    x_new, y_new = splev(u_new, tck, der=0)
    smooth_contour = np.vstack((x_new, y_new)).T.astype(np.int32)
    return smooth_contour.reshape(-1, 1, 2)

# Function to process a frame and draw the inner wall contour
def process_frame_with_contour(frame_path, center, mask, output_path):
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    inner_contour = find_inner_wall_contour(frame_gray, center, mask)

    if inner_contour is not None:
        inner_contour = smooth_contour(inner_contour)  # Smooth the contour
        cv2.drawContours(frame, [inner_contour], -1, (0, 0, 255), 2)  # Draw contour in red

    combined_image = np.hstack((cv2.imread(frame_path), frame))
    cv2.imwrite(output_path, combined_image)

# Main function
def main():
    video_path = "/home/bazzi/TEVG/FSG/Inverse_model/230206_Martini_IVUS_Low_IVC_with_breathing.avi"
    frames_dir = '/home/bazzi/TEVG/FSG/Inverse_model/frames'
    processed_dir = '/home/bazzi/TEVG/FSG/Inverse_model/processed_frames'
    max_frames = 10

    # Step 1: Read the video and create frames
    read_video_and_create_frames(video_path, frames_dir, max_frames)

    # Step 2: Manually select the center of the vessel
    first_frame_path = os.path.join(frames_dir, 'frame_0000.png')
    center = select_center(first_frame_path)
    if center is None:
        print("Error: Center not selected.")
        return

    probe_radius = 27  # Adjust this radius as needed

    # Step 3: Create a mask to exclude the probe region
    first_frame = cv2.imread(first_frame_path, cv2.IMREAD_GRAYSCALE)
    mask = create_probe_mask(center, probe_radius, first_frame.shape)
    cv2.imwrite('mask.png', mask)  # Save the mask for visualization

    # Step 4: Process frames to find and highlight the inner wall contour
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for frame_file in sorted(os.listdir(frames_dir)):
        if frame_file.endswith('.png'):
            frame_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(processed_dir, frame_file)
            remove_probe(frame_path, center, mask, output_path)

    # Display the first processed frame
    processed_frame_path = os.path.join(processed_dir, 'frame_0000.png')
    processed_frame = cv2.imread(processed_frame_path)
    plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
