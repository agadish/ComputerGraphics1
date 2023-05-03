import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    # Assume that size(src) == size(mask).
    pad_width_b = int(center[0] - im_src.shape[0]/2)
    pad_height_b = int(center[1] - im_src.shape[1]/2)
    pad_width_a = int(im_tgt.shape[0] - pad_width_b - im_src.shape[0])
    pad_height_a = int(im_tgt.shape[1] - pad_height_b - im_src.shape[1])


    # padding the source and the mask relative to the center, so that: size(src)==size(mask)==size(target)
    im_src_pad = np.pad(im_src, [(pad_width_b, pad_width_a), (pad_height_b, pad_height_a),(0,0)], mode='constant',constant_values=(0,0))
    im_mask_pad = np.pad(im_mask, [(pad_width_b, pad_width_a), (pad_height_b, pad_height_a)], mode='constant',constant_values=(0,0)) 

    # flattening the images
    f_im_src = im_src_pad.reshape(-1,3)
    f_im_tgt = im_tgt.reshape(-1,3)
    f_im_mask = im_mask_pad.flatten()

    # 1d array, representing the region of the source (=1) and the padding (=0)
    f_im_src_bool_pad = np.pad(np.ones_like(im_src), [(pad_width_b, pad_width_a), (pad_height_b, pad_height_a),(0,0)], mode='constant',constant_values=(0,0)).reshape(-1,3)  

    # same dimensions for all images
    n = im_src_pad.shape[0]
    m = im_src_pad.shape[1]

    # generating D (for the poisson equasion, D*f=b), and B (to calculate the laplacian of the source, B*S=b)
    B_bool = f_im_src_bool_pad[:,0]
    B = laplacian_matrix(n, m, B_bool)
    D = laplacian_matrix(n, m, f_im_mask)

    # b = T (target) outside the region of the mask, and laplacian(S) (source) inside.
    b_bool = np.tile(f_im_mask.reshape(1,-1).transpose(), (1, 3))
    b = np.where(b_bool,B.dot(f_im_src),f_im_tgt)

    # solving the linear equation
    x = spsolve(D, b)
    x = np.where(x>255,255,x) # overflow check
    x = np.where(x<0,0,x) # overflow check
    #x = cv2.threshold(x, 0, 255, cv2.THRESH_TRUNC)[1]
    im_tgt = x.reshape((n,m,3)).astype(np.uint8)
    im_blend = im_tgt
    return im_blend

#n=len(rows),m=len(col),omega=flat mask, 0 outside, or 1 inside. returning D=sparsed laplacian matrix depends of the region
def laplacian_matrix(n, m, omega): 
    N = n*m

    # if the pixel is outside the region (omega==0), we demand: f=T
    A = scipy.sparse.eye(N)#, dtype=np.uint8)

    # if the pixel is inside the region (omega==1), we demand: f=laplacian(S)
    indxs = np.where(np.logical_or(omega==1,omega==255))[0]
    row = indxs
    col = indxs
    data = np.ones_like(row)*(-5)
    A += scipy.sparse.csr_matrix((data, (row, col)), shape=(N,N))
    row = []
    col = []
    for idx in indxs:
        i = idx//m
        j = idx%m
        if j<m-1:
            row.append(idx)
            col.append(idx+1)
        if j>0:
            row.append(idx)
            col.append(idx-1)
        if i<n-1:
            row.append(idx)
            col.append(idx+m)
        if i>0:
            row.append(idx)
            col.append(idx-m)
    row = np.array(row)
    col = np.array(col)        
    data = np.ones_like(row)
    tmp = scipy.sparse.csr_matrix((data, (row, col)), shape=(N,N))
    A += tmp
    return A

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[0] / 2), int(im_tgt.shape[1] / 2)) # opposite? 1,0

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)
    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
