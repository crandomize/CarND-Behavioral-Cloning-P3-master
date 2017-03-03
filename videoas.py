from moviepy.editor import ImageSequenceClip
import argparse
import os
import csv


def loadData(locations, side='center'): 
    samples = []
    for dirname in locations:
        csvfilename = './data/' + dirname + '/driving_log.csv'
        pref = './data/' + dirname + '/IMG/'
        with open(csvfilename) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                #as folder may be moved and file paths are absolute
                line[0] = pref+line[0].split('\\')[-1]
                line[1] = pref+line[1].split('\\')[-1]
                line[2] = pref+line[2].split('\\')[-1]
                
                #if(float(line[3])==0):
                #    continue
                if(side=='center'):
                    samples.append(line[0])
                elif (side=='right'):
                    samples.append(line[1])
                elif (side=='left'):
                    samples.append(line[2])
    
    return samples






def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--side',
        type=str,
        default='center',
        help='center, left, right.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    train_samples = loadData([args.image_folder], side=args.side)
    #print(train_samples)
    video_file = './data/' + args.image_folder + '.' + args.side + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(train_samples, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
