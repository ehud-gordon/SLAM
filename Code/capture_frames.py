import os
import cv2

def resize_image(im, res):
    res = int(res)
    if res==1080:
        return cv2.resize(im, (1920, 1080), interpolation=cv2.INTER_AREA)
    elif res==720:
        return cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)
    elif res==480:
        return cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA)
    elif res==240:
        return cv2.resize(im, (352, 240), interpolation=cv2.INTER_AREA)

def resize_image_specific(im, width, height):
    return cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)

def resize_folder(old_dirpath, new_dirpath, new_res):
    os.makedirs(new_dirpath, exist_ok=True)
    for filename in os.listdir(old_dirpath):
        file_path = os.path.join(old_dirpath, filename)
        im = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        im = resize_image(im=im, res=new_res)
        resized_im_path = os.path.join(new_dirpath, filename)
        cv2.imwrite(filename=resized_im_path, img=im)

def get_frames_dirpath(video_filename, res):
    basename = str(os.path.basename(video_filename).split(".")[0])
    new_dir_path = os.path.join('extracted_frames', basename + "_" + str(res))
    os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path

def extract_frames(video_filename, res=None):
    """ Extracts frames from video, saves them at ./extracted_frames/video_name
    :param video_filename: relative or full path of video file """
    frames_dirpath = get_frames_dirpath(video_filename=video_filename, res=res)
    vidcap = cv2.VideoCapture(video_filename)
    success, image = vidcap.read()
    count = 0
    while success:
        if res != "original":
            image = resize_image(im=image, res=res)
        # save frame as png file
        new_filename = os.path.join(frames_dirpath, 'frame{}.png'.format(count))
        cv2.imwrite(new_filename, image)
        # read new frame
        success, image = vidcap.read()
        count += 1
    print('{}: Read {} frames'.format(video_filename, count))

def main():
    video_paths_to_capture = ["videos\\mine.mp4"]
    res = "480" # could be "original" / "240' / "480" / "720" / "1080"
    for video_filename in video_paths_to_capture:
        # This'll save extracted frames in folder "extracted_frames"
        extract_frames(video_filename=video_filename, res=res)

if __name__ == '__main__':
    # resize_folder('cal_images_untouched', 'cal_images_untouched_480', '480')
    main()