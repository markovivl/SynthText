import numpy as np
import cv2


def get_text_seg(mask):
    seg = mask.astype(np.uint8)
    kernel_dilate = np.ones((25, 25), dtype=np.uint8)
    kernel_erode = np.ones((15, 15), dtype=np.uint8)
    seg = cv2.dilate(seg, kernel_dilate)
    seg = cv2.erode(seg, kernel_erode).astype(np.uint8)
    seg[seg > 0] = 255
    return seg


def unify_coords(all_char):
    x_coords = all_char[0][:, 0].reshape(-1, 4).T
    y_coords = all_char[0][:, 1].reshape(-1, 4).T
    charbb = np.array([x_coords, y_coords])
    return charbb


def split_coords(all_char):
    return np.array(list(zip(all_char[0].T.flatten(), all_char[1].T.flatten())), np.float32)


def transform_charbb(bb, spline_transformer):
    all_char = split_coords(bb)
    all_char = all_char[None, :, :]
    _, spline_char = spline_transformer.applyTransformation(all_char)
    transformed_char_bb = unify_coords(spline_char)
    return transformed_char_bb


def cut_word(num, wordbb, mask):
    x_coords = wordbb[0, :, num].astype(np.float32)
    y_coords = wordbb[1, :, num].astype(np.float32)
    min_x, max_x = np.min(x_coords).astype(np.int_), np.max(x_coords).astype(np.int_)
    min_y, max_y = np.min(y_coords).astype(np.int_), np.max(y_coords).astype(np.int_)
    min_y = max(min_y - 2, 0)
    min_x = max(min_x - 2, 0)
    max_y = min(max_y + 2, mask.shape[0])
    max_x = min(max_x + 2, mask.shape[1])
    word_mask = mask[min_y:max_y, min_x:max_x]
    cut_mask = np.zeros(mask.shape)
    cut_mask[min_y:max_y, min_x:max_x] = word_mask
    cut_mask = cut_mask.astype(np.float32)
    return cut_mask, x_coords, y_coords


def _sin_transform(x_coords_dir, y_coords_dir, d=0, p=1):
    max_x = np.max(x_coords_dir)
    min_x = np.min(x_coords_dir)

    norm_y = -np.sin(p * ((x_coords_dir - min_x)/(max_x - min_x)) * np.pi + d * np.pi)
    unnorm_y = (norm_y - d * np.pi) / np.pi * (max_x - min_x)
    new_y_coords_dir = unnorm_y + abs(np.min(unnorm_y)) + min(y_coords_dir)
    return new_y_coords_dir


def correct_y(y_coords, mask):
    if np.min(y_coords) < 0:
        y_coords -= np.min(y_coords)
    elif np.max(y_coords) > mask.shape[0]:
        y_coords -= (np.max(y_coords) - mask.shape[0])
    return y_coords


def sin_transform(cut_mask, x_coords, y_coords, d=0, p=None):
    x_coords_up = np.linspace(x_coords[0], x_coords[1], 100).astype(np.float32)
    y_coords_up = np.linspace(y_coords[0], y_coords[1], 100).astype(np.float32)
    x_coords_down = np.linspace(x_coords[3], x_coords[2], 100).astype(np.float32)
    y_coords_down = np.linspace(y_coords[2], y_coords[3], 100).astype(np.float32)

    if abs(x_coords[0] - x_coords[1]) > 120:
        p = 0.8
    elif abs(x_coords[0] - x_coords[1]) <= 60:
        p = 0.4
    elif abs(x_coords[0] - x_coords[1]) > 60:
        p = 0.6
            
    start_x_coords = np.hstack((x_coords_up, x_coords_down))
    start_y_coords = np.hstack((y_coords_up, y_coords_down))
    sshape = np.array(list(zip(start_x_coords, start_y_coords)))
    
    new_y_coords_down = _sin_transform(x_coords_down, y_coords_down, d=d, p=p)
    new_y_coords_up = _sin_transform(x_coords_up, y_coords_up, d=d, p=p)
    
    
    new_x_coords = np.hstack((x_coords_up, x_coords_down))
    new_y_coords = np.hstack((new_y_coords_up, new_y_coords_down))
    
    new_y_coords = correct_y(new_y_coords, cut_mask)
    
    if np.max(new_y_coords) > cut_mask.shape[0] or np.min(new_y_coords) < 0:
        return sshape, sshape
    
    ttshape = np.array(list(zip(new_x_coords, new_y_coords)))
    
    
    
    return sshape, ttshape


def spline_transform(sshape, ttshape, mask):
    splines = cv2.createThinPlateSplineShapeTransformer()
    sshape = np.array(sshape.reshape(1,-1,2), dtype=np.float32)
    ttshape = np.array(ttshape.reshape(1,-1,2), dtype=np.float32)
    matches = list()
    for i in range(sshape.shape[1]):
        matches.append(cv2.DMatch(i,i,0))

    splines.estimateTransformation(ttshape, sshape,  matches)
    ret, tshape_ = splines.applyTransformation(sshape)
    warpedimage=splines.warpImage(mask)
    splines.estimateTransformation(sshape, ttshape, matches)
    return warpedimage, splines


def _arc_transform(x, y):
    min_x = min(x)
    max_x = max(x)
    ans = min(y) + ((max_x - min_x)/2) * (-np.sqrt(1 - (((2 * x - min_x - max_x) / (max_x - min_x)) ** 2)))
    return ans


def arc_transform(cut_mask, x_coords, y_coords):
    x_coords_up = np.linspace(x_coords[0], x_coords[1], 100).astype(np.float32)
    y_coords_up = np.linspace(y_coords[0], y_coords[1], 100).astype(np.float32)
    x_coords_down = np.linspace(x_coords[3], x_coords[2], 100).astype(np.float32)
    y_coords_down = np.linspace(y_coords[2], y_coords[3], 100).astype(np.float32)
    
    
def circle_transform_down(cut_mask, x_coords, y_coords):
    x_coords_up = np.linspace(x_coords[0], x_coords[1], 100).astype(np.float32)
    y_coords_up = np.linspace(y_coords[0], y_coords[1], 100).astype(np.float32)
    x_coords_down = np.linspace(x_coords[3], x_coords[2], 100).astype(np.float32)
    y_coords_down = np.linspace(y_coords[2], y_coords[3], 100).astype(np.float32)

    start_x_coords = np.hstack((x_coords_up, x_coords_down))
    start_y_coords = np.hstack((y_coords_up, y_coords_down))
    sshape = np.array(list(zip(start_x_coords, start_y_coords)))

    min_x = np.min(x_coords_down)
    max_x = np.max(x_coords_down)

    min_y = np.min(y_coords_down)
    rad = ((max_x - min_x) / (2 * (np.pi)))

    norm_x = (x_coords_down - min_x)/(max_x - min_x) * 1.9 * np.pi
    new_x = np.cos(norm_x) * rad
    new_y = np.sin(norm_x) * rad

    new_x_down = (new_x - np.min(new_x))/(np.max(new_x) - np.min(new_x))
    new_x_down = (new_x_down * (max_x - min_x)/(0.9 *np.pi) + min_x + (max_x - min_x)/np.pi) 

    new_y_down = new_y /(np.max(new_y) - np.min(new_y))
    new_y_down = new_y_down  * (np.max(new_x_down) - np.min(new_x_down)) + min_y

    #scale_new_y = correct_y(scale_new_y)

    new_x_up = (new_x_down - np.mean(new_x_down)) * 2 + np.mean(new_x_down)
    new_y_up = (new_y_down - np.mean(new_y_down)) * 2 + np.mean(new_y_down)
    
    new_x_coords = np.hstack((new_x_up, new_x_down))
    new_x_coords = new_x_coords
    new_y_coords = np.hstack((new_y_up, new_y_down))
    
    new_y_coords = correct_y(new_y_coords, cut_mask)
    
    ttshape = np.array(list(zip(new_x_coords, new_y_coords)))
    
    return sshape, ttshape


def circle_transform_up(cut_mask, x_coords, y_coords):
    x_coords_up = np.linspace(x_coords[0], x_coords[1], 100).astype(np.float32)
    y_coords_up = np.linspace(y_coords[0], y_coords[1], 100).astype(np.float32)
    x_coords_down = np.linspace(x_coords[3], x_coords[2], 100).astype(np.float32)
    y_coords_down = np.linspace(y_coords[2], y_coords[3], 100).astype(np.float32)

    start_x_coords = np.hstack((x_coords_up, x_coords_down))
    start_y_coords = np.hstack((y_coords_up, y_coords_down))
    sshape = np.array(list(zip(start_x_coords, start_y_coords)))

    min_x = np.min(x_coords_up)
    max_x = np.max(x_coords_up)

    min_y = np.min(y_coords_up)
    rad = ((max_x - min_x) / (2 * (np.pi)))

    norm_x = (x_coords_up - min_x)/(max_x - min_x) * 1.9 * np.pi
    new_x = np.cos(norm_x) * rad
    new_y = np.sin(norm_x) * rad

    new_x_up = (new_x - np.min(new_x))/(np.max(new_x) - np.min(new_x))
    new_x_up = (new_x_up * (max_x - min_x)/(0.9 *np.pi) + min_x + (max_x - min_x)/np.pi) 

    new_y_up = new_y /(np.max(new_y) - np.min(new_y))
    new_y_up = new_y_up  * (np.max(new_x_up) - np.min(new_x_up)) + min_y

    #scale_new_y = correct_y(scale_new_y)

    new_x_down = (new_x_up - np.mean(new_x_up)) * 2 + np.mean(new_x_up)
    new_y_down = (new_y_up - np.mean(new_y_up)) * 2 + np.mean(new_y_up)

    
    new_x_coords = np.hstack((new_x_up, new_x_down))
    new_x_coords = cut_mask.shape[1] - new_x_coords
    new_y_coords = np.hstack((new_y_up, new_y_down))
    
    new_y_coords = correct_y(new_y_coords, cut_mask)
    
    ttshape = np.array(list(zip(new_x_coords, new_y_coords)))
    
    return sshape, ttshape


def clean_circle(ttshape, new_img):
    x_start = int(ttshape[100, 0])
    x_end = int(ttshape[99, 0])
    y_start = int(ttshape[99, 1])
    y_end = int(ttshape[100, 1])
    x_start, x_end = min(x_start, x_end), max(x_start, x_end)
    y_start, y_end = min(y_start, y_end), max(y_start, y_end)
    y_mid = (y_start + y_end)//2
    x_mid = (x_start + x_end)//2
    #new_img[y_start - 4:(y_end + 4), x_start - 2:x_end + 2] = 0
    new_img[y_mid - 5:y_mid + 5, x_mid - 7:x_mid + 7] = 0
    return new_img
    
def translate(tx=0, ty=0):
    T = np.eye(3)
    T[0:2,2] = [tx, ty]
    return T

def scale(s=1, sx=1, sy=1):
    T = np.diag([s*sx, s*sy, 1])
    return T

def rotate(degrees, y_max):
    T = np.eye(3)
    # just involves some sin() and cos()
    T[0:2] = cv2.getRotationMatrix2D(center=(0, y_max), angle=-degrees, scale=1.0)
    return T

def rotate_and_scale(ttshape, new_img, angle=90, scale_f=1):
    min_coords = np.min(ttshape, axis=0)
    max_coords = np.max(ttshape, axis=0)
    y_max, x_max = new_img.shape
    rotation_point = min_coords + (max_coords - min_coords)/2
    icx, icy = rotation_point
#     scale_f = min(scale_f, min(new_img.shape[::-1] / (max_coords - min_coords)))
#     rotation_point = min_coords + (max_coords - min_coords)/2
#     hippo = np.sqrt(np.sum(np.abs(max_coords - min_coords)**2))
#     if scale_f * abs(np.sin(angle/57.3)) * hippo  > y_max:
#         scale_f = min(scale_f, y_max / hippo)
#     if scale_f * abs(np.cos(angle/57.3)) * hippo > x_max:
#         scale_f = min(scale_f, x_max / hippo)
    
    T1 = translate(-icx, y_max - icy)
    T2 = rotate(-angle, y_max)
    T3 = scale(scale_f)
    T4 = translate(x_max/2, (-y_max * scale_f + y_max/2))
    Tf = T4 @ T3 @ T2 @ T1
    M = Tf[0:2]
    rotated_img = cv2.warpAffine(new_img, M,
                                 new_img.shape[::-1], flags=cv2.INTER_LINEAR)
    return rotated_img    
    
