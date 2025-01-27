from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader
import cv2
import math
from scipy.spatial.transform import Rotation
import copy
import yaml

def test_cases():
    # ================ Test find_idx_of_nearest() ================
    array = [0, 1, 2, 3, 4, 5]
    np.testing.assert_equal(3, find_idx_of_nearest(array, 3))

    array = [0, 1, 2, 3, 4, 5, 7]
    np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    with np.testing.assert_raises(ValueError):
        array = [7, 0, 1, 2, 3, 4, 5, 7]
        np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    with np.testing.assert_raises(ValueError):
        array = [0, 1, 1, 1, 2, 3, 4, 5, 7]
        np.testing.assert_equal(1, find_idx_of_nearest(array, 1.3))

    # ============== Test load_images_and_timestamps() ==========
    bag_path = 'scenario1_robotA'
    path_to_dataset = Path(Path.cwd(), 'dataset_AirMuseum_Seq1', bag_path, bag_path[10:], 'original')
    path_to_bag_100 = Path(path_to_dataset, 'cam100_imu.bag')
    topic_name = '/robotA/cam100/image_raw'
    timestamps, timestamps_num, images = load_images_and_timestamps(path_to_bag_100, topic_name, 3, 5)
    
    np.testing.assert_equal(len(timestamps), 3)
    np.testing.assert_equal(len(timestamps_num), 3)
    np.testing.assert_equal(len(images), 3)
    np.testing.assert_equal(1566474939.199000000, timestamps_num[0])
    np.testing.assert_equal(1566474939, timestamps[0].sec)
    np.testing.assert_equal(199000000, timestamps[0].nanosec)
    image_des_20 = [14, 12, 14, 13, 15, 13, 12, 11, 12, 11, 12, 11, 10, 10, 10, 11, 10, 11, 9, 11]
    np.testing.assert_array_equal(image_des_20, images[0][0,:20])

def calculate_avg_time_diff(ts_num_1, ts_num_2):
    total = 0
    for i in range(len(ts_num_1)):
        total += (np.abs(ts_num_1[i] - ts_num_2[i]))
    return total/len(ts_num_1)

def find_idx_of_nearest(array, value):
    # Make sure the array is sorted
    if not np.all(array[:-1] <= array[1:]):
        raise ValueError("Input array is not sorted")
    
    # Make sure that there are no duplicate values
    unique = np.unique(array)
    if len(unique) != len(array):
        raise ValueError("Array has duplicate entries")

    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array)) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx]):
        return idx-1
    else:
        return idx

def load_images_and_timestamps(bag_path, topic_name, num_images, starting_index):
    # Arrays
    timestamps = []
    timestamps_num = []
    images = []

    # Create reader instance and open for reading.
    i = 0
    with AnyReader([bag_path]) as reader:
        connections = [x for x in reader.connections if x.topic == topic_name]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            if starting_index is not None and i < starting_index:
                i += 1
                continue
            if num_images is not None and i >= num_images + starting_index:
                break
            msg = reader.deserialize(rawdata, connection.msgtype)
            timestamps.append(msg.header.stamp)
            timestamps_num.append(msg.header.stamp.sec + (msg.header.stamp.nanosec / (10**9)))
            image_data = np.reshape(np.array(msg.data, dtype=np.uint8), (512, 640))
            images.append(image_data)
            i += 1

    return timestamps, timestamps_num, images

def extract_images_and_ts(seq_path, bag_path, topic_name, path_name, calib_file, visualize_flow=False, visualize_disparity=True):
    print("============ Extracting Sequence: ", bag_path, " ============")

    # Set entries to save
    num_dataset_entries = 100
    starting_index = 400

    # Set up a reader to read the rosbags and GT
    path_to_dataset = Path(Path.cwd(), seq_path, bag_path, bag_path[10:], 'original')
    path_to_bag_100 = Path(path_to_dataset, 'cam100_imu.bag')
    path_to_bag_101 = Path(path_to_dataset, 'cam101.bag')
    path_to_gt_pose_original = Path(path_to_dataset, "cam100_stamped_groundtruth.txt").absolute()

    # Set up paths to save files
    path_to_save_R_images = Path(Path.cwd(), 'datasets', path_name, 'image_R').absolute()
    path_to_save_L_images = Path(Path.cwd(), 'datasets', path_name, 'image_L').absolute()
    path_to_save_L_flow = Path(Path.cwd(), 'datasets', path_name, 'flow_L').absolute()
    path_to_gt_pose_vdo = Path(Path.cwd(), 'datasets', path_name, "pose_gt.txt").absolute()
    path_to_times_file = Path(Path.cwd(), 'datasets', path_name, 'times.txt').absolute()

    # Load in calibration yaml files
    path_to_calib = Path(Path.cwd(), seq_path, 'sensors', calib_file)
    yaml_file = None
    with open(str(path_to_calib)) as stream:
        yaml_file = yaml.safe_load(stream)

    # Calculate camera intrinsic matrices
    params_L = yaml_file["cam1"]["intrinsics"]
    intrins_L = np.array([  [params_L[0],            0, params_L[2]], 
                            [          0,  params_L[1], params_L[3]],
                            [          0,            0,           1]])
    params_R = yaml_file["cam0"]["intrinsics"]
    intrins_R = np.array([  [params_R[0],            0, params_R[2]], 
                            [          0,  params_R[1], params_R[3]],
                            [          0,            0,           1]])
    
    # Load distortion coefficients
    DL = np.array(yaml_file["cam1"]["distortion_coeffs"])
    DR = np.array(yaml_file["cam0"]["distortion_coeffs"])

    # Load the camera 1 (left) to 0 (right) transformation
    # This is what we need for Stereo Rectify: https://stackoverflow.com/questions/28678985/exact-definition-of-the-matrices-in-opencv-stereorectify
    H_camL_to_camR = np.array(yaml_file["cam1"]["T_cn_cnm1"])
    R_camL_to_camR = H_camL_to_camR[0:3,0:3]
    T_camL_to_camR = np.array(H_camL_to_camR[0:3,3]).transpose()

    # Calculate the camera 0 to camera 1 transformation
    H_camR_to_camL = np.linalg.inv(H_camL_to_camR)

    # Read in all the pose lines
    pose_ts = []
    pose_data = []
    with open(path_to_gt_pose_original, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_split = line.split(" ")
            pose_ts.append(line_split[0])
            rest = line_split[1:]
            pose_data.append(rest)
    pose_ts = np.array(pose_ts)

    # Remove comments from pose data
    pose_ts = pose_ts[1:]
    pose_data = pose_data[1:]
    pose_ts_num = copy.deepcopy(pose_ts).astype(np.float128)

    # Convert to floats
    for i in range(len(pose_ts)):
        pose_ts[i] = float(pose_ts[i])

    for i in range(len(pose_data)):
        for j in range(len(pose_data[i])):
            pose_data[i][j] = float(pose_data[i][j])

    # Keep only poses we want
    pose_ts = np.array(pose_ts[starting_index:starting_index+num_dataset_entries])
    pose_data = np.array(pose_data[starting_index:starting_index+num_dataset_entries]).astype(np.float128)
    pose_ts_num = pose_ts_num[starting_index:starting_index+num_dataset_entries]
    np.testing.assert_equal(num_dataset_entries, len(pose_ts))

    # Write the pose_gt.txt file
    with open(path_to_gt_pose_vdo, "w") as f:
        # Iterate through each pose
        for i in range(0, len(pose_ts)):        
            # Calculate the H matrix for this translation and quaternion
            T_camR = np.array([pose_data[i][0:3]]).transpose()
            R_camR = Rotation.from_quat(np.array(pose_data[i][3:])).as_matrix()
            H_camR = np.concatenate((R_camR, T_camR), axis=1)
            H_camR = np.concatenate((H_camR, np.array([[0, 0, 0, 1]])), axis=0)

            # This H is for the right camera, but we need it for the left
            H_camL = H_camR_to_camL @ H_camR

            # Write the result
            line = str(i)
            for row in H_camL:
                for val in row:
                    line += " " + str(val)
            line += "\n"
            f.write(line)

    # Load images and timestamps for both cameras
    ts_L, ts_num_L, distorted_images_L = load_images_and_timestamps(path_to_bag_101, topic_name + "/cam101/image_raw", None, None)
    ts_R, ts_num_R, distorted_images_R = load_images_and_timestamps(path_to_bag_100, topic_name + "/cam100/image_raw", None, None)

    # Calculate mappings to undistort/rectify stereo images
    image_shape = distorted_images_R[0].shape
    image_shape = (image_shape[1], image_shape[0]) # Need to flip image shape, probably due to numpy vs. opencv dimension differences
    RL, RR, PL, PR, Q = cv2.fisheye.stereoRectify(intrins_L, DL, intrins_R, DR, image_shape, R_camL_to_camR, T_camL_to_camR, cv2.fisheye.CALIB_ZERO_DISPARITY)
    mapL_x, mapL_y = cv2.fisheye.initUndistortRectifyMap(intrins_L, DL, RL, PL, image_shape, cv2.CV_32FC1)
    mapR_x, mapR_y = cv2.fisheye.initUndistortRectifyMap(intrins_R, DR, RR, PR, image_shape, cv2.CV_32FC1)
    newKL, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(PL)
    newKR, _, _, _, _, _, _ = cv2.decomposeProjectionMatrix(PR)
    print("New Camera Left Projection Matrix: ", newKL)
    print("New Camera Right Projection Matrix: ", newKR)
    print("")

    # Keep track of used images
    used_images_L_idx = []
    used_images_R_idx = []
    used_images_gt_ts = []
    used_images_L_ts = []
    used_images_R_ts = []
    images_L = []
    images_R = []

    # Define the number of lines to draw
    num_lines = 10
    line_color = (0, 255, 0)  # Green lines
    line_thickness = 1

    # Write the closest camera 100 and camera 101 images to each pose
    for i in range(0, len(pose_ts_num)):
        # Get closest timestamp index for left camera
        idxL = find_idx_of_nearest(ts_num_L, pose_ts_num[i])
        used_images_L_ts.append(ts_num_L[idxL])

        # Get closest timestamp index for right camera
        idxR = find_idx_of_nearest(ts_num_R, pose_ts_num[i])
        used_images_R_ts.append(ts_num_R[idxR])

        # Make sure we haven't already used either of these images
        if idxL in used_images_L_idx:
            raise AssertionError("This left camera image is closest to 2 GT poses! Something is wrong.")
        if idxR in used_images_R_idx:
            raise AssertionError("This right camera image is closest to 2 GT poses! Something is wrong.")
        used_images_L_idx.append(idxL)
        used_images_R_idx.append(idxR)

        used_images_gt_ts.append(pose_ts_num[i])

        # Undistort & rectify the stereo images
        assert mapL_x.shape == distorted_images_L[idxL].shape[:2], "Mapping dimensions mismatch"
        assert mapR_x.shape == distorted_images_R[idxR].shape[:2], "Mapping dimensions mismatch"
        image_L_rectified = cv2.remap(distorted_images_L[idxL], mapL_x, mapL_y, cv2.INTER_CUBIC)
        image_R_rectified = cv2.remap(distorted_images_R[idxR], mapR_x, mapR_y, cv2.INTER_CUBIC)

        # Write both images
        image_name = str(i).rjust(6, '0') + ".jpeg"
        if not cv2.imwrite(Path(path_to_save_R_images, image_name), image_R_rectified):
            print("Could not write image")
        if not cv2.imwrite(Path(path_to_save_L_images, image_name), image_L_rectified):
            print("Could not write image")
        images_L.append(image_L_rectified)
        images_R.append(image_R_rectified)

        # Draw horizontal lines
        # height, width = image_L_rectified.shape[:2]
        # for i in range(1, num_lines + 1):
        #     y = i * height // (num_lines + 1)
        #     cv2.line(image_L_rectified, (0, y), (width, y), line_color, line_thickness)
        #     cv2.line(image_R_rectified, (0, y), (width, y), line_color, line_thickness)

        # # Concatenate images side by side for visualization
        # combined_image = np.hstack((image_L_rectified, image_R_rectified))

        # # Display the combined image
        # cv2.imshow('Rectified Stereo Images with Lines', combined_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    # Validate the the error is low enough
    print("(Camera 1) Average Time Sync Difference to GT: ", calculate_avg_time_diff(used_images_gt_ts, used_images_R_ts))
    print("Standard Deviation: ", np.std(np.array(used_images_gt_ts) - np.array(used_images_R_ts)))
    print("(Camera 2) Average Time Sync Difference to GT: ", calculate_avg_time_diff(used_images_gt_ts, used_images_L_ts))
    print("Standard Deviation: ", np.std(np.array(used_images_gt_ts) - np.array(used_images_L_ts)))
    print("Images taken every 0.05 seconds (20 Hertz) and GT Pose every 0.1 seconds (10 Hz). Make sure timestamp error above is reasonable.\n")

    # Write the timestamps to a file
    with open(path_to_times_file, "w") as f:
        for i, stamp in enumerate(pose_ts):
            time_str = str(stamp)
            if i < len(pose_ts) - 1:
                time_str += "\n"
            f.write(time_str)

    # Calculate dense optical flow using OpenCV
    for i in range(len(used_images_L_idx) - 1):
        image_prev = images_L[i]
        image_next = images_L[i+1]

        # Calculate dense optical flow using Farneback's method
        flow = cv2.calcOpticalFlowFarneback(
            prev=image_prev,
            next=image_next,
            flow=None,
            pyr_scale=0.5,  # Image scale (<1) to build pyramids
            levels=3,       # Number of pyramid layers
            winsize=15,     # Averaging window size
            iterations=3,   # Number of iterations at each pyramid level
            poly_n=5,       # Size of pixel neighborhood
            poly_sigma=1.2, # Gaussian standard deviation
            flags=0         # Operation flags (0 for default)
        )

        # Save the calculated flow
        flow_name = str(i).rjust(6, '0') + ".flo"
        if not cv2.writeOpticalFlow(str(Path(path_to_save_L_flow, flow_name)), flow):
            print("Could not write Optical Flow")

        # Visualize the optical flow (convert flow to HSV)
        if visualize_flow:
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(cv2.cvtColor(image_prev, cv2.COLOR_GRAY2BGR))
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue: direction
            hsv[..., 1] = 255                      # Saturation: full
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude
            rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            #Display the result
            cv2.imshow('Dense Optical Flow', rgb_flow)
            cv2.waitKey(0)

    # Calculate disparity map using OpenCV StereoSGBM
    # for i in range(len(used_images_L_idx)):
    #     image_left = images_L[i]
    #     image_right = images_R[i]

    #     # Create a StereoBM object and compute the disparity map
    #     block_size = 25
    #     stereo = cv2.StereoSGBM_create(
    #         minDisparity=0,
    #         numDisparities=16*10,
    #         blockSize=block_size,
    #         P1=8 * 1 * (block_size ** 2),
    #         P2=32 * 1 * (block_size ** 2),
    #         disp12MaxDiff=1,
    #         uniquenessRatio=15,
    #         speckleWindowSize=100,
    #         speckleRange=2,
    #         mode=cv2.StereoSGBM_MODE_HH
    #     )
    #     disparity = stereo.compute(image_left,image_right)
    #     #disparity = np.clip(disparity, 0, 255)

    #     # Normalize the disparity map for visualization
    #     if visualize_disparity:
    #         import matplotlib.pyplot as plt
    #         disparity = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #         cv2.imshow('Dense Optical Flow', disparity)
    #         cv2.waitKey(0)

    # cv2.destroyAllWindows()

    # Calculate Baseline distance for Settings file
    baseline = np.sqrt(T_camL_to_camR[0]**2 + \
                       T_camL_to_camR[1]**2 + \
                       T_camL_to_camR[2]**2)
    print("Baseline: ", baseline)

def main():
    # Run test cases
    test_cases()

    # NOTE: At least for robot A, cam100 is right eye, cam101 is left eye
    extract_images_and_ts('dataset_AirMuseum_Seq1', 'scenario1_robotA', '/robotA', 'AirMuseum-Seq1-A', 'robotA_cameras_calib.yaml')
    extract_images_and_ts('dataset_AirMuseum_Seq1', 'scenario1_robotB', '/robotB', 'AirMuseum-Seq1-B', 'robotB_cameras_calib.yaml')

    # Currently, disparity is calulated using mobilestereonet.
    # Also, Semantic Segmentation is done by CAT-SEG, but sadly it's NOT Instance Segementation, so this needs to be replaced.

if __name__ == "__main__":
    main()