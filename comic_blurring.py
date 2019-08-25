import cv2
import os
from tqdm import tqdm

"""
Tis function is from https://github.com/nijuyr/comixGAN
"""
def smooth_image_edges(img, plot=False):
    # Get edges
    edges = cv2.Canny(img,30,60)

    # Dilate edges with kernel (5,5) with 15 iterations
    dilated_edges = cv2.dilate(edges,(7,7), iterations=25)
    dilated_edges_to_compare = dilated_edges.copy()
    dilated_edges_to_compare[dilated_edges == 0] = -1

    # Copy image twice
    img_no_dilated_edges, img_only_dilated_edges = img.copy(), img.copy()

    # Prepare images with region of only edges and no edges
    img_no_dilated_edges[dilated_edges_to_compare != -1] = 0
    img_only_dilated_edges[dilated_edges_to_compare == -1] = 0
    
    # Gaussian blur of the image with region of only edges
    blurred_edges = cv2.GaussianBlur(img_only_dilated_edges,(9,9),0)
    
    # Clip to take only region of edges (without values blurred on the remaining parts of the image)
    blurred_edges[dilated_edges_to_compare == -1] = 0

    # Final Gaussian blur of sum of images with and without edges
    result = blurred_edges + img_no_dilated_edges
    result = cv2.GaussianBlur(result,(9,9),0)
    
    if plot:
        plt.figure(figsize=(16,10))
        plt.subplot(221),plt.imshow(img[:,:,[2,1,0]])
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(222),plt.imshow(edges, cmap = 'gray')
        plt.title('Edges'), plt.xticks([]), plt.yticks([])
        plt.subplot(224),plt.imshow(dilated_edges, cmap = 'gray')
        plt.title('Dilated edges'), plt.xticks([]), plt.yticks([])
        plt.subplot(223),plt.imshow(result[:,:,[2,1,0]])
        plt.title('Blurred Image'), plt.xticks([]), plt.yticks([])
        plt.show()        
    return result

if __name__ == '__main__':
    samples = os.listdir(os.path.join('datasets', 'comic'))

    for sample in tqdm(samples):
        img = cv2.imread(os.path.join('datasets', 'comic', sample))
        res = smooth_image_edges(img)
        cv2.imwrite(os.path.join('datasets', 'comic_blurred', sample), res)