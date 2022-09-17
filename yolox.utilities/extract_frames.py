import numpy as np
import os
import cv2
import logging
import argparse

def collect_videos(path):

        files = []

        #check the path
        if (os.path.isdir(path)):
            
            for video in sorted(os.listdir(path)):
                complete_path = os.path.join(path, video)
                if complete_path.upper().endswith((".MP4", ".mp4")):
                    
                    logging.info('* collecting video: ' + video + ' from folder: ' + path)
                    files.extend([{
                        'path': complete_path,
                        'name': os.path.splitext(video)[0],
                        'hour_folder': path
                    } 
                    ])
        elif (os.path.isfile(path)):
            if path.upper().endswith((".MP4", ".mp4")):
            
                logging.info('* collecting video: ' + path)
                root = os.path.splitext(path)[0]
                files.extend([{
                        'path': path,
                        'name': root.split('/')[-1],
                    } 
                    ])
            else:
                raise ("File is not a video. Must End with: .MP4, .mp4")

        else:
        
            raise ('Path should be a directory or a video file')

    
        return files

def main():

    for file in collect_videos(args.file):
        #with open(file['path'], "rb") as video:
    
            capture = cv2.VideoCapture(file['path'])
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
            fps = int(capture.get(cv2.CAP_PROP_FPS))
            frames_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            every = fps // args.fps
            index = 0
            saved_count = 0
            while index < frames_number:
                #Capture frame-by-frame
                ret, frame = capture.read()
                if not ret:
                    break           
                if frame is None:
                    continue
                
                if index % every == 0:
                    # arr = np.ndarray((frame.size, 1), buffer=frame, dtype=np.uint8)
                    # print(f"frame: {index} frame shape: {arr.shape} sent to pipeline")
                    save_path = os.path.join(args.output, "{:3d}.jpg".format(saved_count))  # create the save path
                    print ('Creating...' + save_path)
                    if args.resize_dim:
                        if isinstance(args.resize_dim, int):
                            dim = [args.resize_dim, args.resize_dim]
                        elif isinstance(args.resize_dim, list):
                            if len(args.resize_dim) == 2:
                                dim = args.resize_dim
                            else:
                                raise("dimension list must be in a form [M, N]")

                        frame = cv2.resize(frame, dim)
        
                    cv2.imwrite(save_path, frame)  # save the extracted image
                    saved_count += 1

                index += 1
                if index == 100:
                    break

            logging.info('Video name * ' + file['name'] +' * with FPS: '+str(fps)+ '. Need fps: '+str(args.fps)+' sending every '+str(every)+' frames')
            logging.info('Processed ' +str(saved_count) + ' frames from ' + str(index))

            # When everything done, release the video capture object
            capture.release()
            # Closes all the frames
            cv2.destroyAllWindows()

if __name__ == '__main__':

    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, help='Video path')
    parser.add_argument('-fps', type=int, default= None, help='set fps number for extraction')
    parser.add_argument('-resize_dim', type=(int,list), help='dimension to resize extracted frames. Input a single Integer for N*N dimension or input a dimension list [N,M]')
    parser.add_argument('-output', type=str, help='output path to save extracted frames')
    args = parser.parse_args()

    #files = collect_videos(args.file)
    main()
