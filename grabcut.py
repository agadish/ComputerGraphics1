import numpy as np
import cv2
import argparse
import sklearn.mixture
import scipy.stats
from igraph import Graph

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel


# Define the GrabCut algorithm function
def grabcut(img, rect, n_components=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[(rect[1]+rect[3])//2, (rect[0]+rect[2])//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask, n_components=n_components)

    num_iters = 1000
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM, n_components=n_components)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

class MyGMM(object):
    def __init__(self, n_components):
        self._n_components = n_components
    
    def calc_dist_weights(self, values):
        # TODO: calcualte by ourselves
        mixture = sklearn.mixture.GaussianMixture(self.n_components, init_params='kmeans').fit(values)
        self._mixture = mixture
        self._dists = [scipy.stats.multivariate_normal(mixture.means_[i], mixture.covariances_[i])
                       for i in range(self.n_components)]
    
    @property
    def dists(self):
        return self._dists
    
    @property
    def means(self):
        return self._mixture.means_
    
    @property
    def covariances(self):
        return self._mixture.covariances_
    
    @property
    def n_components(self):
        return self._n_components
    
    @property
    def weights(self):
        return self._mixture.weights_
    
    # def get_beta(self, img):
        # flat_img = img.flatten()
        # avg_square_distance = np.mean([)
        # beta = np.linalg.inv(2 * (dist ** 2))



def initalize_GMMs(img, mask, n_components=5):
    # fgVals = img[(GC_FGD == mask) + (GC_PR_FGD == mask)]
    # bgVals = img[(GC_BGD == mask) + (GC_PR_BGD == mask)]

    fgGMM = MyGMM(n_components)
    # fgGMM.fit(fgVals)

    bgGMM = MyGMM(n_components)
    # bgGMM.fit(bgVals)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM, n_components=5):
    fgVals = img[(GC_FGD == mask) + (GC_PR_FGD == mask)]
    bgVals = img[(GC_BGD == mask) + (GC_PR_BGD == mask)]
    bgGMM.calc_dist_weights(bgVals)
    fgGMM.calc_dist_weights(fgVals)   

    # bgDist, bgWeighst = bgGMM.calc_dist_weights(bgVals)
    # fgDist, fgWeights = fgGMM.calc_dist_weights(fgVals)

    # # TODO: implement GMM component assignment step
    # choose_gaussian = lambda rgb: np.argmax(fgGMM)  
    # for i in range(K):
    #     mean = bgGMM.means_[i,:]
    #     cov = bgGMM.covariances_[i,:,:]
    #     bg_cov = img[mask]
    #     icov = np.linalg.inv(cov)
    #     w = bgGMM.weights_[i]
    #     deter = np.linalg.det(cov)

    return bgGMM, fgGMM

def calculate_N(img, m, n, beta=0.5):
    dist = np.linalg.norm(img[m] - img[n])
    result = 50 / np.linalg.norm(np.array(m) - np.array(n)) * np.exp(-beta * (dist ** 2))
    return result

def calculate_D(z, means, covs, weights, n_components=5):
    result = -np.log(sum(
        [weights[i] / np.sqrt(np.linalg.det(covs[i]) *
                              np.exp(0.5 * np.matmul(np.matmul((z - means[i]).reshape(1, -1), np.linalg.inv(covs[i])), z - means[i])))
        for i in range(n_components)]
    ))[0]
    return result

def mask_equals(m1, m2):
    bg_group = (GC_BGD, GC_PR_BGD, )
    fg_group = (GC_FGD, GC_PR_FGD, )
    result = False
    if m1 in bg_group:
        result = m2 in bg_group
    else:
        result = m2 in fg_group

    return result

def get_neighbours(img, mask, m):
    all_neighbours = list()
    i,j = m
    if j > 0:
        all_neighbours.append((i, j - 1, ))
    if i > 0:
        all_neighbours.append((i - 1, j, ))
    if i < img.shape[0] - 1:
        all_neighbours.append((i + 1, j, ))
    if j < img.shape[1] - 1:
        all_neighbours.append((i, j + 1, ))

    # neighbours = [n for n in all_neighbours if mask_equals(mask[m], mask[n])]
    neighbours = all_neighbours
    return neighbours

def calculate_K(img, mask, beta=0.5):
    # TODO: all edges, or depending on the type?
    return np.max([sum([calculate_N(img, m, n, beta) for n in get_neighbours(img, mask, m)])
                   for m in np.ndindex(img.shape[:-1])])

def calculate_beta(img):
    return 0.5 # TODO

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0
    
    # Build graph
    img_f = img.flatten()
    mask_f = mask.flatten()
    beta = calculate_beta(img)
    K_val = calculate_K(img, mask, beta)
    g = Graph()
    g.add_vertex(name='source')
    g.add_vertex(name='sink')
    g.add_vertices(img_f)
    def index_to_edges(z):
        if GC_BGD == mask[z]:
            return [('source', z, K_val)]
        if GC_FGD == mask[z]:
            return [(z, 'sink', K_val)]
        if mask[z] in (GC_PR_BGD, GC_PR_FGD, ):
            bg_edge = ('source', z, calculate_D(img[z], bgGMM.means, bgGMM.covariances, bgGMM.weights, bgGMM.n_components))
            fg_edge = (z, 'sink', calculate_D(img[z], fgGMM.means, fgGMM.covariances, fgGMM.weights, fgGMM.n_components))
            return [bg_edge, fg_edge]
        raise TypeError(f'Unknown mask type at {z}')

    edges_sets = [index_to_edges(z) for z in np.ndindex(img.shape[:-1])]
    edges = [e for e in s for s in edges_sets]
    g.add_adges(edges)
    min_cut = g.mincut()
    energy = min_cut.energy
    # U = sum([calculate_D(rgb, bgGMM.means, bgGMM.covariances, bgGMM.weights, bgGMM.n_components) for rgb in img.flatten()])
    
    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
