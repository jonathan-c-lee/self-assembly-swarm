""" Generate video or clean video data. """
import os


if __name__ == '__main__':
    command = input('Create video? [y/N]: ')
    if command == 'y':
        frame_rate = 24
        image_path = "'./images/tmp_%d.png'"
        video_name = './videos/video.mp4'
        os.system(f'ffmpeg -framerate {frame_rate} -i {image_path} {video_name}')
    elif command != 'N':
        print('Invalid input')
    else:
        command = input('Delete images? [y/N]: ')
        if command == 'y':
            os.system('rm -rf images/*.png')
        elif command != 'N':
            print('Invalid input')

