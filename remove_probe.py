import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from scipy.interpolate import splprep, splev

# Function to read the video and create frames
def read_video_and_create_frames(video_path, output_dir, max_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

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
    return frame_idx

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

def draw_spline_contours(frame):
    points = []

    def draw_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Draw Points", frame)
    
    cv2.namedWindow("Draw Points")
    cv2.setMouseCallback("Draw Points", draw_points)
    cv2.imshow("Draw Points", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) > 3:
        points = np.array(points)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        return spline_points
    else:
        return np.array(points)
    
def draw_spline_contours(frame):
    points = []

    def draw_points(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Draw Points", frame)
    
    cv2.namedWindow("Draw Points")
    cv2.setMouseCallback("Draw Points", draw_points)
    cv2.imshow("Draw Points", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) > 3:
        points = np.array(points)
        tck, u = splprep([points[:, 0], points[:, 1]], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        return spline_points
    else:
        return np.array(points)
    
# Main function
def main():
    video_path = "/home/bazzi/TEVG/FSG/Inverse_model/230206_Martini_IVUS_Low_IVC_with_breathing.avi"
    frames_dir = '/home/bazzi/TEVG/FSG/Inverse_model/frames'
    processed_dir = '/home/bazzi/TEVG/FSG/Inverse_model/processed_frames'
    contours_dir = '/home/bazzi/TEVG/FSG/IVUS-processing/contours'
    max_frames = 3

    # Step 1: Read the video and create frames
    frame_count = read_video_and_create_frames(video_path, frames_dir, max_frames)
    if frame_count == 0:
        return

    probe_radius = 26  # Adjust this radius as needed

    
    # Step 4: Process frames to remove the probe
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    contours_list = []
    for frame_file in sorted(os.listdir(frames_dir)):
        if frame_file.endswith('.png'):

            # Step 2: Manually select the center of the vessel
            probe_frame_path = os.path.join(frames_dir, frame_file)
            center = select_center(probe_frame_path)
            if center is None:
                print("Error: Center not selected.")
                return
            
            # Step 3: Create a mask to exclude the probe region
            probe_frame = cv2.imread(probe_frame_path, cv2.IMREAD_GRAYSCALE)
            mask = create_probe_mask(center, probe_radius, probe_frame.shape)
            cv2.imwrite('mask.png', mask)  # Save the mask for visualization
            frame_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(processed_dir, frame_file)
            remove_probe(frame_path, center, mask, output_path)

            # Display the processed frame
            processed_frame_path = os.path.join(processed_dir, frame_file)
            processed_frame = cv2.imread(processed_frame_path)
            plt.imshow(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
    i =0;        
    for frame_file in sorted(os.listdir(frames_dir)):
        print(f"Draw inner contour for frame {frame_file}")
        frame_path_processed = os.path.join(processed_dir, frame_file)
        frame_processed = cv2.imread(frame_path_processed)
        inner_contour = draw_spline_contours(frame_processed.copy())
        contours_list.append(inner_contour)
        # Save contours in .noy formart
        np.save(os.path.join(contours_dir, f'inner_contour_{i:04d}.npy'), inner_contour)
        
        # Create masks based on the drawn contours
        inner_mask = np.zeros_like(frame_processed[:, :, 0])
        if len(inner_contour) > 0:
            cv2.drawContours(inner_mask, [inner_contour], -1, 255, -1)

         # Debugging: Show the masks
        cv2.imshow(f"Inner Mask - Frame {i}", inner_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Debugging: Ensure masks are correctly applied
        masked_frame = cv2.bitwise_and(frame_processed, frame_processed, mask=inner_mask)
        cv2.imshow(f"Masked Frame - Frame {i}", masked_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        i+=1




    
    

if __name__ == "__main__":
    main()
