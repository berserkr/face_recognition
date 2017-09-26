import cv2
import argparse

def process_image(image_path):
    image = face_recognition.load_image_file(image_path)

    # need to apply gabor filter 
    



parser = argparse.ArgumentParser(description='Get cancellable bloom filter from image')
parser.add_argument('-b', '--bloom',
            action="store", dest="bloom",
            help="Get bloom filter", default="bloom")

args = parser.parse_args()
print (args.bloom)



