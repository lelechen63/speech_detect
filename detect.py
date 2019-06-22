import dlib
import cv2
import numpy as np
import os
import argparse

# print(cv2.__version__)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i','--in_file', type=str, default='./example.avi')
	return parser.parse_args()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
config = parse_args()



video_name = config.in_file



def _extract_audio(video_name):
   
	command = 'ffmpeg -i ' + video_name + ' -ar 16000  -ac 1  ' + './example.wav' 
	try:
	    # pass
	    os.system(command)
	except BaseException:
	    print (line)

def face_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)        
        shapes.append(shape)

    return shapes, rects
def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

def norm_landmark( img_shape, shape ):
	shape = shape * 1.0 /  img_shape

	return shape 

def judge( current_shape):
	mouth_region = current_shape[49:]
	##### step 1: check if the mouth is open or not
	d1 = current_shape[67,1]  - current_shape[61,1]
	d2 = current_shape[66,1]  - current_shape[62,1]
	d3 = current_shape[65,1]  - current_shape[63,1]
	total_d = d1 + d2 + d3
	print (d1, d2, d3, total_d)
	return total_d
def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

      
def image_to_video(sample_dir = None, video_name = None):
    
    command = 'ffmpeg -framerate 25  -i ' + sample_dir +  '/%04d.jpg -c:v libx264 -y -vf format=yuv420p ' + video_name 
    print (command)
    os.system(command)

def add_audio(video_name=None, audio_dir = None):

    command = 'ffmpeg -i ' + video_name  + ' -i ' + audio_dir + ' -vcodec copy  -acodec copy -y  ' + video_name.replace('.mp4','.mov')

    print (command)
    os.system(command)
def main(video_name):
	vidcap = cv2.VideoCapture(video_name)
	success,image = vidcap.read()
	count = 0
	success = True
	privios_d = 1.0

	small_counter = 0
	while success:
		print (count)
		img = image.copy()
		# print (image.shape)
		img_shape = image.shape[:-1]
		faces , rects = face_detect(image)

		if len(faces) > 1:
			sys.exit('Error! Can not deal with one face in a video. If we want to deal with multiple faces, we need to associate faces between frames.')

		face = faces[0]
		# print (face.shape)
		face = norm_landmark(img_shape,  face)
		current_d = judge(face)

		if current_d < 0.03:
			small_counter += 1
			if small_counter > 4:
				color = (0,0,255)
			else:
				color = (255,0,0)
		else:
			small_counter = 0
			color = (0,255,0)

		cv2.rectangle(img,(rects[0].left(),rects[0].top()),(rects[0].right(),rects[0].bottom()),color,3)
		cv2.imwrite('./result/%04d.jpg'%count, img)
		cv2.imshow('%d'%count,img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		success,image = vidcap.read()
		count += 1
	vidcap.release()
	cv2.destroyAllWindows()


	_extract_audio( video_name = video_name)
	image_to_video( sample_dir = './result', video_name = './result.mp4')

	add_audio(video_name='./result.mp4', audio_dir = './example.wav')


main(video_name)