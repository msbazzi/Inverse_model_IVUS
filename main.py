import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import xml.etree.ElementTree as ET

# Function to draw spline contours manually
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
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck, der=0)
        spline_points = np.vstack((x_new, y_new)).T.astype(np.int32)
        return spline_points
    else:
        return np.array(points)

# Function to calculate radial displacement
def calculate_radial_displacement(inner_contours):
    displacements = []
    for i in range(1, len(inner_contours)):
        prev_contour = inner_contours[i - 1]
        curr_contour = inner_contours[i]

        if len(prev_contour) == len(curr_contour):
            distances = np.sqrt((prev_contour[:, 0] - curr_contour[:, 0]) ** 2 +
                                (prev_contour[:, 1] - curr_contour[:, 1]) ** 2)
            displacements.append(distances)
        else:
            print(f"Contours do not match in frame {i-1} and frame {i}")
    
    average_displacement = np.mean(displacements, axis=0)
    return average_displacement

def main():
    # Load the IVUS video file
    video_path = "/home/bazzi/TEVG/FSG/Inverse_model/230206_Martini_IVUS_Low_IVC_with_breathing.avi"
    # Directory to save extracted frames and contours
    frames_dir = '/home/bazzi/TEVG/FSG/Inverse_model/frames'
    contours_dir = '/home/bazzi/TEVG/FSG/Inverse_model/contours'
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(contours_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Process only the first 10 frames
    max_frames = 10
    frame_idx = 0

    inner_contours = []

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save the frame
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
        cv2.imwrite(frame_path, frame)

        # Draw spline contours for inner wall
        print(f"Draw inner contour for frame {frame_idx}")
        inner_contour = draw_spline_contours(frame.copy())
        inner_contours.append(inner_contour)
        
        # Save contours in .npy format
        np.save(os.path.join(contours_dir, f'inner_contour_{frame_idx:04d}.npy'), inner_contour)

        frame_idx += 1

    cap.release()
    print(f'Processed and saved contours for {frame_idx} frames.')

    # Calculate radial displacement
    radial_displacement = calculate_radial_displacement(inner_contours)
    
    # Save radial displacement
    np.savetxt('/home/bazzi/TEVG/FSG/Inverse_model/radial_displacement.csv', radial_displacement, delimiter=',')
    print(f'Saved radial displacement for {len(radial_displacement)} points.')

    # Visualize the displacement
    plt.plot(radial_displacement)
    plt.xlabel('Contour Point Index')
    plt.ylabel('Radial Displacement (pixels)')
    plt.title('Radial Displacement of Inner Vessel Wall')
    plt.show()

if __name__ == "__main__":
    main()
