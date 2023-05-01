import numpy as np
import cv2
import argparse
import sklearn.mixture
import scipy.stats
from igraph import Graph
import itertools
np.warnings.filterwarnings('ignore')

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel
SOURCE_NODE = 'source'
SINK_NODE = 'sink'
EPSILON = 0.00001

import time

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
        t1 = time.time()
        print(f'[iter {i}] Running update_GMMs....')
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        t2 = time.time()
        print(f'[iter {i}] Took {(t2-t1):.1f}sec. Running calculate_mincut...')

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        t3 = time.time()
        print(f'[iter {i}] Took {(t3-t2):.1f}sec, energy {energy:.1f}. Running update_mask...')

        mask = update_mask(mincut_sets, mask)
        t4 = time.time()
        print(f'[iter {i}] Took {(t4-t3):.1f}sec. Running checking convergence...')

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

class NLinks(object):
    def __init__(self, img):
        self._img = img
        self._beta = None
        self._k = None
        self._all_edges = None
        self._weights = None
        self._N_for_directions = None
        self._down_square_diff = None
        self._right_square_diff = None
        self._downright_square_diff = None
        self._downleft_square_diff = None

    @property
    def down_edges(self):
        h = self._img.shape[0]
        w = self._img.shape[1]
        uppers = np.arange((h - 1) * w)
        lowers = uppers + w
        ul = np.vstack((uppers, lowers, )).T
        return ul.tolist()
    
    @property
    def right_edges(self):
        h = self._img.shape[0]
        w = self._img.shape[1]
        all_indexes = np.arange(h * w)
        lefts = all_indexes[all_indexes % w != w - 1]
        rights = lefts + 1
        ul = np.vstack((lefts, rights, )).T
        return ul.tolist()
        
    @property
    def downright_edges(self):
        h = self._img.shape[0]
        w = self._img.shape[1]
        all_indexes = np.arange(h * w)
        uplefts = all_indexes[np.logical_and(all_indexes % w != w - 1, all_indexes < (h - 1) * w)]
        downrights = uplefts + (w + 1)
        ul = np.vstack((uplefts, downrights, )).T
        return ul.tolist()        
   
    @property
    def downleft_edges(self):
        h = self._img.shape[0]
        w = self._img.shape[1]
        all_indexes = np.arange(h * w)
        uprights = all_indexes[np.logical_and(all_indexes % w != 0, all_indexes < (h - 1) * w)]
        downlefts = uprights + (w - 1)
        ul = np.vstack((uprights, downlefts, )).T
        return ul.tolist()
    
    @property
    def down_square_diff(self):
        """
        @brief |z_m - z_n|^2 only for down neighbours
        """
        if self._down_square_diff is None:
            self._down_square_diff = np.sum(np.square(self._img[:-1, :] - self._img[1:, :]), axis=-1)
        return self._down_square_diff

    @property
    def right_square_diff(self):
        """
        @brief |z_m - z_n|^2 only for right neighbours
        """
        if self._right_square_diff is None:
            self._right_square_diff = np.sum(np.square(self._img[:, :-1] - self._img[:, 1:]), axis=-1)
        return self._right_square_diff

    @property
    def downright_square_diff(self):
        """
        @brief |z_m - z_n|^2 only for downright neighbours
        """
        if self._downright_square_diff is None:
            self._downright_square_diff = np.sum(np.square(self._img[:-1, :-1] - self._img[1:, 1:]), axis=-1)
        return self._downright_square_diff


    @property
    def downleft_square_diff(self):
        """
        @brief |z_m - z_n|^2 only for downleft neighbours
        """
        if self._downleft_square_diff is None:
            self._downleft_square_diff = np.sum(np.square(self._img[:-1, 1:] - self._img[1:, :-1]), axis=-1)
        return self._downleft_square_diff

    @property
    def beta(self):
        if self._beta is None:
            self._beta = self._calculate_beta()

        return self._beta
    
    def _calculate_beta(self):
        all_square_diffs = (self.down_square_diff,
                            self.right_square_diff,
                            self.downright_square_diff,
                            self.downleft_square_diff, )
        expected_distance_square_with_factor = np.sum([np.sum(s) for s in all_square_diffs])
        # 4 * h * w - 3 * (h + w) + 2
        factor = 4 * self._img.shape[0] * self._img.shape[1] - 3 * (self._img.shape[0] + self._img.shape[1]) + 2
        expected_distance_square = expected_distance_square_with_factor / factor    # 4hw - 3w -3h + 2
        if expected_distance_square == 0:
            expected_distance_square = 0.00001
        beta = 1 / (2 * expected_distance_square)
        return beta
    
    def _N_from_square_diffs_matrix(self, square_diff_matrix, distance):
        return 50 / distance * np.exp(self.beta * square_diff_matrix)
    
    @property
    def N_for_directions(self):
        if self._N_for_directions is None:
            N_down = self._N_from_square_diffs_matrix(self.down_square_diff, 1)
            N_right = self._N_from_square_diffs_matrix(self.right_square_diff, 1)
            N_downright = self._N_from_square_diffs_matrix(self.downright_square_diff, np.sqrt(2))
            N_downleft = self._N_from_square_diffs_matrix(self.downleft_square_diff, np.sqrt(2)) 
            self._N_for_directions = (N_down, N_right, N_downright, N_downleft, )
        
        return self._N_for_directions

    def _calculate_K(self):
        # 1. Get 4 basic N-links values
        N_down, N_right, N_downright, N_downleft = self.N_for_directions
        
        # 2. Replicate the 4 basic to all the 8 N-links
        N_link_sum_per_pixel = np.zeros(self._img.shape[:2])
        # Up-left
        N_link_sum_per_pixel[1:, 1:] += N_downright
        # Up
        N_link_sum_per_pixel[1:, :] += N_down
        # Up-right
        N_link_sum_per_pixel[1:, :-1] += N_downleft
        # Left
        N_link_sum_per_pixel[:, :-1] += N_right
        # Right
        N_link_sum_per_pixel[:, 1:] += N_right
        # Down-left
        N_link_sum_per_pixel[:-1, 1:] += N_downleft
        # Down
        N_link_sum_per_pixel[:-1, :] += N_down
        # Down-right
        N_link_sum_per_pixel[:-1, :-1] += N_downright

        # 3. Get the value of the member with the highest N-links score
        K = np.max(N_link_sum_per_pixel)

        return K

    @property
    def K(self):
        # Maximize all the N values
        if self._k is None:
            self._k = self._calculate_K()
        return self._k

    @property
    def edges(self):
        if self._all_edges is None:
            # Order is down, right, downright, downleft
            self._all_edges = self.down_edges + self.right_edges + self.downright_edges + self.downleft_edges
        return self._all_edges
    
    def _calculate_weights(self):
        all_edges = list(itertools.chain(*[n.flatten().tolist() for n in self.N_for_directions]))
        return all_edges
    
    @property
    def weights(self):
        """
        Weights of the edges, in the same order that retured in edges
        """
        if self._weights is None:
            self._weights = self._calculate_weights()
        return self._weights
    
def requires_mask(f):
    def new_f(self, *args, **kwargs):
        if self._mask_f is None or self._orig_mask_shape is None:
            raise Exception('Function requires mask')
        return f(self, *args, **kwargs)
    return new_f

def requires_K(f):
    def new_f(self, *args, **kwargs):
        if self._K is None:
            raise Exception('Function requires K')
        return f(self, *args, **kwargs)
    return new_f

class TLinks(object):
    def __init__(self, img):
        self._orig_image_shape = img.shape
        self._img_f = img.reshape(-1, 3)
        self._orig_mask_shape = None
        self._mask_f = None
        self._K = None
        self._edges = None

    def update_mask(self, mask):
        self._orig_mask_shape = mask.shape
        self._mask_f = mask.flatten()

    def set_K(self, K):
        self._K = K

    def calculate_D(self, z, means, dets, icovs, weights, n_components=5):
        result = -np.log(np.sum(
            [weights[i] / np.sqrt(
                dets[i] * np.exp(0.5 * np.matmul(np.matmul((z - means[i]).reshape(1, -1), icovs[i]),
                                                 z - means[i])))
            for i in range(n_components)]
        ))
        return result

    @requires_mask
    def index_to_weights(self, z, bgGMM, fgGMM):
        mask_f = self._mask_f
        img_f = self._img_f
        if GC_BGD == mask_f[z]:
            return [self._K]
        if GC_FGD == mask_f[z]:
            return [self._K]
        if mask_f[z] in (GC_PR_BGD, GC_PR_FGD, ):
            bg_weight = self.calculate_D(img_f[z], fgGMM.means_, fgGMM.determinants_, fgGMM.icovariances_, fgGMM.weights_, fgGMM.n_components)
            fg_weight = self.calculate_D(img_f[z], bgGMM.means_, bgGMM.determinants_, bgGMM.icovariances_, bgGMM.weights_, bgGMM.n_components)
            return [bg_weight, fg_weight]
        raise TypeError(f'Unknown mask type at {z}')

    @requires_mask
    def index_to_edges(self, z):
        mask_f = self._mask_f
        img_f = self._img_f
        if GC_BGD == mask_f[z]:
            return [(SOURCE_NODE, z)]
        if GC_FGD == mask_f[z]:
            return [(z, SINK_NODE)]
        if mask_f[z] in (GC_PR_BGD, GC_PR_FGD, ):
            return [(SOURCE_NODE, z), (z, SINK_NODE)]
        raise TypeError(f'Unknown mask type at {z}')
    
    @property
    def edges(self):
        mask_f = self._mask_f
        if self._edges is None:
            dual_edges = [self.index_to_edges(z) for z in range(self._img_f.shape[0])]
            self._edges = list(itertools.chain(*dual_edges))
        return self._edges
    
    def get_weights(self, bgGMM, fgGMM):
        weights_sets = [self.index_to_weights(z, bgGMM, fgGMM) for z in range(self._img_f.shape[0])]
        weights = list(itertools.chain(*weights_sets))
        return weights
    
class Grabcut(object):
    def __init__(self, img):
        self._nlinks = NLinks(img)
        self._tlinks = TLinks(img)
        self._tlinks.set_K(self._nlinks.K)
    
    def calculate_edges_weights(self, mask, bgGMM, fgGMM):
        self._tlinks.update_mask(mask)
        edges = self._nlinks.edges + self._tlinks.edges
        weights = self._nlinks.weights + self._tlinks.get_weights(bgGMM, fgGMM)
        return edges, weights
    
    @property
    def edges(self):
        return self._nlinks.edges + self._tlinks.edges
    
    @property
    def weights(self):
        return self._nlinks.weights + self._tlinks.weights
    
    def update_mask(self, mask):
        self._tlinks.update_mask(mask)
    

class MyGMM(object):
    def __init__(self, n_components):
        self._n_components = n_components
        self._nlinks = None
        self._icovariances = None
        self._determinants = None
        self._data_dim = 3
    
    def fit(self, values):
        # TODO: calcualte by ourselves
        self._n_components = min(self._n_components, len(values))
        if self._n_components == 1:
            values = np.repeat(values, 2, axis=0)
            
        mixture = sklearn.mixture.GaussianMixture(self.n_components, init_params='kmeans').fit(values)
        self._mixture = mixture
        self._dists = [scipy.stats.multivariate_normal(mixture.means_[i], mixture.covariances_[i])
                       for i in range(self.n_components)]
    
    @property
    def dists(self):
        return self._dists
    
    @property
    def means_(self):
        return self._mixture.means_
    
    @property
    def covariances_(self):
        if 1 == self.n_components:
            return np.identity(self._data_dim) * EPSILON
        return self._mixture.covariances_
        
    @property
    def icovariances_(self):
        if self._icovariances is None:
            self._icovariances = np.linalg.inv(self.covariances_)
        return self._icovariances
        
    @property
    def determinants_(self):
        if self._determinants is None:
            self._determinants = np.linalg.det(self.covariances_)
        return self._determinants
    
    @property
    def n_components(self):
        return self._n_components
    
    @property
    def weights_(self):
        return self._mixture.weights_

g_should_exit = False
g_grabcut = None
def initalize_GMMs(img, mask, n_components=5):
    global g_grabcut
    g_grabcut = Grabcut(img)

    fgGMM = MyGMM(n_components)
    bgGMM = MyGMM(n_components)

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    fgVals = img[(GC_FGD == mask) + (GC_PR_FGD == mask)]
    bgVals = img[(GC_BGD == mask) + (GC_PR_BGD == mask)]
    
    if len(bgVals) <= 1:
        bgGMM = None
    else:
        bgGMM.fit(bgVals)

    if len(fgVals) <= 1:
        fgGMM = None
    else:
        fgGMM.fit(fgVals)   

    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    global g_grabcut
    min_cut = [[], []]
    energy = 0
    # Build graph
    if not bgGMM or not fgGMM:
        return min_cut, energy
    
    g = Graph()
    g.add_vertices(range(img.shape[0] * img.shape[1]))
    g.add_vertex(name=SOURCE_NODE)
    g.add_vertex(name=SINK_NODE)
    edges, weights = g_grabcut.calculate_edges_weights(mask, bgGMM, fgGMM)
    g.add_edges(edges)
    min_cut_ext = g.st_mincut(SOURCE_NODE, SINK_NODE, weights)
    source_sink_indexes = [g.vs.find(SOURCE_NODE).index, g.vs.find(SINK_NODE).index]
    min_cut = tuple([i for i in cut if i not in source_sink_indexes] for cut in min_cut_ext)
    energy = min_cut_ext.value
    
    return min_cut, energy

def update_mask(mincut_sets, mask):
    if not np.any(mincut_sets):
        global g_should_exit
        g_should_exit = True
        return mask
    
    old_mask = mask
    mask_f = mask.flatten()
    condition__gc_pr_bgd = np.logical_and(np.isin(np.arange(mask_f.size), mincut_sets[0]), mask_f == GC_PR_FGD)
    condition__gc_pr_fgd = np.logical_and(np.isin(np.arange(mask_f.size), mincut_sets[1]), mask_f == GC_PR_BGD)
    mask = np.where(condition__gc_pr_bgd, GC_PR_BGD,
                    np.where(condition__gc_pr_fgd, GC_PR_FGD, mask_f)).reshape(mask.shape)
    global g_grabcut
    g_grabcut.update_mask(mask)
    print(f'Number of mask changes (prev {np.sum(old_mask)} new {np.sum(mask)}: {np.sum(np.nonzero(old_mask - mask))}')
    return mask

STATIC_ENERGY_CONVERGENCE_COMBO = 5
STATIC_ENERGY_THRESHOLD = 2000
g_static_energy_combo = 0
g_prev_energy = None
def check_convergence(energy):
    global g_should_exit
    if not energy or g_should_exit:
        return True
    
    global g_static_energy_combo
    global g_prev_energy
    
    if g_prev_energy is not None:
        if np.abs(energy - g_prev_energy) < STATIC_ENERGY_THRESHOLD:
            g_static_energy_combo += 1
        else:
            g_static_energy_combo = 0
    
    g_prev_energy = energy
    convergence = g_static_energy_combo >= STATIC_ENERGY_CONVERGENCE_COMBO

    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation
    if not gt_mask.size:
        accuracy = 1
    else:
        accuracy = np.sum(predicted_mask != gt_mask) / gt_mask.size

    predicted_intersection_count = np.sum(np.logical_and(predicted_mask == gt_mask, (predicted_mask == GC_PR_FGD) + (predicted_mask == GC_FGD)))
    total_predictable = np.sum((predicted_mask == GC_PR_FGD) + (predicted_mask == GC_FGD) + (gt_mask == GC_PR_FGD) + (gt_mask == GC_FGD))
    if not total_predictable:
        jaccard_value = 1
    else:
        jaccard_value = predicted_intersection_count / total_predictable
    return accuracy, jaccard_value

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

