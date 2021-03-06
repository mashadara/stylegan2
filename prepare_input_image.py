import cv2
import dlib
import numpy as np
import PIL.Image
import scipy.ndimage


_detector = dlib.get_frontal_face_detector()
_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def process_source(in_path, out_path, output_size=1024):
    # Find the largest face in the image
    image = cv2.imread(in_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_rects = _detector(image_gray, 1)
    areas = [rect.area() for rect in face_rects]
    biggest_face_rect = face_rects[areas.index(max(areas))]

    # Get this face's landmarks
    landmarks = _predictor(image_gray, biggest_face_rect)

    # Convert to a np array
    lm = np.zeros((68,2), dtype='float')
    for i in range(68):
        lm[i] = (landmarks.part(i).x, landmarks.part(i).y)

    transform_size = 4096
    enable_padding = True
    # The following is taken from the FFHQ processing code at https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)   # CoM of all eye L points
    eye_right    = np.mean(lm_eye_right, axis=0)  # CoM of all eye R points
    eye_avg      = (eye_left + eye_right) * 0.5   # CoM of all eye points
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Check dimensions
    quad_minside = min(
        np.hypot(*(quad[0,:]-quad[1,:])),
        np.hypot(*(quad[1,:]-quad[2,:])),
        np.hypot(*(quad[2,:]-quad[3,:])),
        np.hypot(*(quad[3,:]-quad[0,:]))
    )
    if quad_minside < output_size*0.5:
        print('Warning: Image resolution is low (%.1f%% of ideal) which may lead to poor results. Consider using a higher-resolution image.' % (quad_minside / output_size * 100))

    # Load the image
    img = PIL.Image.open(in_path)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save
    img.save(out_path)


if __name__ == '__main__':
    import sys
    process_source(sys.argv[1], sys.argv[2])
