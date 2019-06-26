import cv2
import numpy as np
import os
import sys
from samples.coco import coco
from mrcnn import utils
from mrcnn import model as modellib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("background_path", type=str, help="path of the desired background image")
parser.add_argument("person", type=int, default=1, help="Number of person (max) you want to extract from each file(Image/video) / default is 1")
parser.add_argument('-l','--list_of_files', nargs='+', help="path of the input images or videos", required=True)
parser.add_argument("-v", "--video", help="If video file is the input", action="store_true")
args = parser.parse_args()

# Load the pre-trained model data
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Change the config information
class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1
    
config = InferenceConfig()

# COCO dataset object names
model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)

# Resize the image (460, 380)
def do_resize(image):
    size = (460, 380)
    resized_img = cv2.resize(image, size, interpolation = cv2.INTER_AREA)
    return resized_img

# This funtion takes images and extracted person pixel level information (mask) as an input and returns the final merged image after attaching new background
# image[:,:,c_channel] means: image[:,:,0] is the Blue channel,image[:,:,1] is the Green channel, image[:,:,2] is the Red channel
# mask_arr == 1 means that these pixels belongs to the object.
# np.where function work: In the background image, if the pixel belong to object, change it to object-pixel-values to place object into the new background. 
def merge_images_and_apply_new_background(image, mask):
    # desired background image path
    background_image = str(args.background_path)
    # reading the background image
    background_img = cv2.imread(background_image)
    # changing the size equal to (460, 380)
    resized_background_img = do_resize(background_img)
    # Showing the desired background for visulization
    cv2.imshow('Desired_Background', resized_background_img)
    for k in range(len(mask)):
        for i in range(len(mask[k])):
            mask_list = mask[k][i]
            mask_arr = np.array(mask_list)
            for c_channel in range(3):
                resized_background_img[:, :, c_channel] = np.where(
                    mask_arr == 1,
                    image[k][:, :, c_channel],
                    resized_background_img[:, :, c_channel]
                )
    return resized_background_img

# This function takes a list as input and return the first element of it 
def area_element(area_n_index):
    return area_n_index[0]

# This function takes an image and detected object information and return the mask (whick pixel belongs to the person in the given image)
def do_person_identificaiton(image, boxes, masks, ids, desired_n_person_in_a_image = 1):
    # list (area) will save area for all the detected person
    area = []
    # list (mask) will save mask for all the detected person
    mask = []
    # n_instances saves the amount of all objects
    n_instances = boxes.shape[0]

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    # n_person keeps track of total number of person in the image
    n_person = 0
    # loop over all detected objects
    for i in range(n_instances):
        # check if there is a box surrounding the object
        if not np.any(boxes[i]):
            continue
        # checking if object is person => id = 1 means person
        if ids[i] == 1:
            # compute the square of each person-object
            y1, x1, y2, x2 = boxes[i]
            square = (y2 - y1) * (x2 - x1)
            # keep track of the person-body area => inserting into area list [[(area of the body, object id)], ...]
            area.insert(n_person, (square, i))
            n_person+=1
        else:
            continue
    # check if there is no person detected in the image
    if not area:
        desired_n_person_in_a_image = 0
    # check if detected person is less than desired_n_person_in_a_image then extract only detected person otherwise desired_n_person
    elif len(area)<desired_n_person_in_a_image:
        desired_n_person_in_a_image = len(area)
    
    # To extract main person from the image => checked area of the body and here sorting the area 
    sorted_area = sorted(area, reverse=True, key=area_element)
    # from sorted area list, take only first desired_n_person_in_a_image
    for i in range(desired_n_person_in_a_image):
        mask.insert(i, list(masks[:,:,sorted_area[i][1]]))
    
    # Return the input image and mask (identified person pixels)
    return image, mask

# If video is the input
if args.video:
    input_videos = args.list_of_files #["demo/modi_1_Trim.mp4", "demo/ind_Trim.mp4"]
    # objects - to store opencv capture objects of input videos, n_frames - to store total number of frames in each video at frame rate of 25
    objects, n_frames= [[], []]
    # initialize the all video objects
    for i in range(len(input_videos)):
        objects.insert(i, cv2.VideoCapture(input_videos[i]))
    
    # Recording Video 
    fps = 25
    frame_size = (460, 380)
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter("demo_results_merge/saved_video.avi", fcc, fps, frame_size)

    # Calculate the total number of frames in each video file at frame rate of 25
    for i in range(len(input_videos)):
        objects[i].set(5, 25)
        n_frames.insert(i, objects[i].get(7))

    # max no of frames among all video files
    max_frames = max(n_frames)

    # Initilize the frames list of list that will store all frames of each video
    frames = [[] for i in range(int(max_frames))] 
    # Storing the video frames into frames => [[video_1_frames], [video_2_frames], ...]
    for video_counter in range(len(input_videos)):
        for frame_counter in range(int(objects[video_counter].get(7))):
            ret, frame = objects[video_counter].read()
            if not ret:
                break
            frames[frame_counter].insert(video_counter, list(frame))
    # releasing the opencv video capture objests
    for i in objects:
        i.release()
    
    # looping over each video frames.
    # on each frame, 
    # 1. objects will be detected using MASK RCNN algorithm 
    # 2. person object will be detected from function do_person_identificaiton
    # 3. person object will be placed into new background
    # 4. person object from all video frames will be merged by function merge_images_and_apply_new_background
    for i in range(len(frames)):
        image, mask = [[], []]
        for j in range(len(frames[i])):
            cur_frame = np.array(frames[i][j])
            resized_frame = do_resize(cur_frame)
            obj_detection_result = model.detect([resized_frame], verbose=0)
            r = obj_detection_result[0]
            image_r, mask_r = do_person_identificaiton(
                resized_frame, r['rois'], r['masks'], r['class_ids'], args.person
            )
            image.insert(i, image_r)
            mask.insert(i, mask_r)
        
        r_frame = merge_images_and_apply_new_background(image, mask)
        out.write(r_frame)
        cv2.imshow('Output video', r_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cv2.destroyAllWindows()
# if images are the input
else:
    input_image_list = args.list_of_files
    # image -> to store processed images, mask -> to store mask of detected person objects, input_img -> to store input images 
    image, mask, input_img = [[], [], []]
    # on each image, 
    # 1. objects will be detected using MASK RCNN algorithm 
    # 2. person object will be detected from function do_person_identificaiton
    # 3. person object will be placed into new background
    # 4. person object from all images will be merged by function merge_images_and_apply_new_background
    for i in range(len(input_image_list)):
        img = cv2.imread(input_image_list[i])
        input_img.insert(i, cv2.imread(input_image_list[i]))
        resized_img = do_resize(img)
        obj_detection_result = model.detect([resized_img], verbose=0)
        r = obj_detection_result[0]
        image_r, mask_r = do_person_identificaiton(
            resized_img, r['rois'], r['masks'], r['class_ids'], args.person
        )
        image.insert(i, image_r)
        mask.insert(i, mask_r)

    frame = merge_images_and_apply_new_background(image, mask)

    # Showing the Original and Output images for visulization
    for i, val in enumerate(input_img):
        cv2.imshow(f'Input Image {i+1}', val)
    cv2.imshow('Output_Image', frame)

    # Wait for keys to exit (by pressing the esc key) or save
    key = cv2.waitKey(0)
    if key == 27:                 
        cv2.destroyAllWindows()
    elif key == ord('s'):        
        cv2.imwrite('saved_image.jpg', frame)
        cv2.destroyAllWindows()
