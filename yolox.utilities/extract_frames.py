# Importing all necessary libraries
import cv2
import os

# Read the video from specified path
video_path = "./mock_or_ai_1_view_one_chunk0.mp4"
#data path
export_path = "./test_images"
#rescale size if needed
image_size = (608, 608)


def check_path(path):
    if not os.path.exists(path):
        print('video path is not exist. Please create the folder path for videos and place videos there!')
        exit(1)

def extract_frames(video_path, frames_dir, resize=True, start=-1, end=-1, every=1):
    """
    Extract frames from a video using OpenCVs VideoCapture
    :param video_path: path of the video
    :param frames_dir: the directory to save the frames
    :param start: start frame
    :param end: end frame
    :param every: frame spacing
    :return: count of images saved
    """
    
    video_path = os.path.normpath(video_path)  # make the paths OS (Windows) compatible
    frames_dir = os.path.normpath(frames_dir)  # make the paths OS (Windows) compatible

    check_path(frames_dir)
    check_path(video_path)

    capture = cv2.VideoCapture(video_path)  # open the video using OpenCV

    if start < 0:  # if start isn't specified lets assume 0
        start = 0
    if end < 0:  # if end isn't specified assume the end of the video
        end = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.set(1, start)  # set the starting frame of the capture
    frame = start  # keep track of which frame we are up to, starting from start
    while_safety = 0  # a safety counter to ensure we don't enter an infinite while loop (hopefully we won't need it)
    saved_count = 0  # a count of how many frames we have saved

    while frame < end:  # lets loop through the frames until the end

        _, image = capture.read()  # read an image from the capture

        if while_safety > 500:  # break the while if our safety maxs out at 500
            break

        # sometimes OpenCV reads None's during a video, in which case we want to just skip
        if image is None:  # if we get a bad return flag or the image we read is None, lets not save
            while_safety += 1  # add 1 to our while safety, since we skip before incrementing our frame variable
            continue  # skip

        if frame % every == 0:  # if this is a frame we want to write out based on the 'every' argument
            while_safety = 0  # reset the safety count
            save_path = os.path.join(frames_dir, "{:10d}.jpg".format(frame))  # create the save path
            print ('Creating...' + save_path)
            if resize:
                image = cv2.resize(image, image_size)
        
            cv2.imwrite(save_path, image)  # save the extracted image
            saved_count += 1  # increment our counter by one

        frame += 1  # increment our frame count

    capture.release()  # after the while has finished close the capture
    cv2.destroyAllWindows()

    print("{:5d} Images saved".format(saved_count)) # print the count of the images we saved

if __name__ == "__main__":
    #start=-1, end=-1,
    extract_frames(video_path=video_path, frames_dir=export_path, resize=True, every=15)