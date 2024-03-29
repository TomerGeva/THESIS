import numpy as np
import scipy.stats as stats
from scipy.signal import convolve2d
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure


class BlobDetectionFunctions:
    """
        Edge detection is done whenever a laplacian operator over the image produces zero crossing. To make the system
    more noise resistant, we can convolve the image with a gaussian kernel to mean out the noise, over the cost of
    smoothing edges. Therefore, when convolving the image with the Laplacian of Gaussian (LoG) operator, zero crossing
    at the output are the location of the edges in the original image.
        A blob is defined as a "patch" in the image with 2 edges, a rising edge and a descending edge, in proximity.
    When running the LOG operator over a blob with the ideal std, the center of the blob will produce a maxima (since
    the two edges are close, the effects of the operator on each edge are interfering constructively).

    Denoting the Laplacian operator as L(.), we can define the Normalized LoG (NLoG) operator as [sigma^2*L(.)] where the
    sigma matches the sigma of the gaussian kernel. The NLOG operator can be approximated vua using the Difference of
    Guassains (DoG). Noting a gaussian with std sigma as n_{sigma} we get that:

                    n_{s*sigma} - n_{sigma} \aaprox (s-1) * sigma^2+L(n_{sigma}) = (s-1)* NLoG

    This class holds the functions needed to perform blob detection in a grayscale image, i.e. 1 channel image.
    The algorithm works as follows:
    1. create a 3D scale space S(x,y,sigma) via convolving a gaussian kernel with different sigma values with the picture
    2. subtract adjacent maps, creating the DoG result and divide by (s-1) creating the NLoG approximation.
    3. performing 3D local maximum detection to locate the center of the blobs and their scales

    Variables:
        * sigma_0        -> initial scale
        * scale          -> multiplication factor between two adjacent scales
        * k              -> number of scales in the scale space
        * peak_threshold -> threshold for the peak detection
        * kernel_size    -> size of the gaussian kernel
    """
    def __init__(self, peak_threshold=1, kernel_size=25, sigma_0=0.4, scale=1.15, k=10):
        """
        :param peak_threshold: Threshold for local maxima detection
        :param kernel_size: size of the gaussian kernel
        """
        self.peak_th     = peak_threshold
        self.kernel_size = kernel_size
        self.sigma_0     = sigma_0
        self.scale       = scale
        self.k           = k

    @staticmethod
    def create_gaussian_kernel(size, sigma):
        """
        :param size: size of the kernel
        :param sigma: std of the kgaussian
        :return: a 2D square kernel with size "size"
        """
        # =====================================================
        # Local variables
        # =====================================================
        nsig = 3
        x = np.linspace(-nsig, nsig, size)
        kern1d = stats.norm.pdf(x, scale=sigma)
        kern2d = np.outer(kern1d, kern1d)
        return kern2d

    def create_scale_space(self, image):
        """
        :param image:input image, numpy 2D array, meaning 1 channel only
        sigma_0: Initial scale
        scale: The multiplication factor in the scale sense.
        k: size of the scale dimension
        :return: The function computes a scale space from the given 2-D image. This means that:
                    sigma_i = sigma_0 * (scale^i) where i = 0,1, ... , k-1
                For each scale we create a gaussian kernel with std sigma_i and convolve the kernel with the image.
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        sigma_0 = self.sigma_0
        scale   = self.scale
        k       = self.k
        scale_space = np.zeros(list(image.shape) + [k])
        # ==============================================================================================================
        # Performing convolution with the wanted kernels
        # ==============================================================================================================
        for ii in range(k):
            sigma = sigma_0 * (scale**ii)
            kernel = self.create_gaussian_kernel(self.kernel_size, sigma=sigma)
            scale_space[:, :, ii] = convolve2d(image, kernel, mode='same')
        return scale_space

    def create_dog_space(self, scale_space):
        """
        :param scale_space: A 3D scale space  numpy arrat where the scale dimension is at dim=2
        scale: Optional, if not None, dividing the DoG space by (scale-1)
        :return: DoG space, Difference of Gaussian space. This is an approximation for the NLoG space
        """
        scale = self.scale
        sign = -1 if scale > 1 else 1
        dog_space = sign * (scale_space[:, :, 1:] - scale_space[:, :, :-1])
        return dog_space / (scale-1)

    def extract_local_maxima(self, dog_space):
        """
        :param dog_space: Difference of Gaussians 3D space
        :return: function detects local maxima points and returns them as a list of tuples
        """
        # ==============================================================================================================
        # Local variables
        # ==============================================================================================================
        neighborhood = generate_binary_structure(3, 3)
        # ==============================================================================================================
        # Detecting local maxima
        # ==============================================================================================================
        # ----------------------------------------------------------------------------------------------------------
        # Performing max filter and threshold filter
        # ----------------------------------------------------------------------------------------------------------
        local_max = maximum_filter(dog_space, footprint=neighborhood) == dog_space
        th_filter = dog_space > self.peak_th
        # ----------------------------------------------------------------------------------------------------------
        # Isolating maximum from zero surface
        # ----------------------------------------------------------------------------------------------------------
        max_3d = local_max & th_filter
        # ----------------------------------------------------------------------------------------------------------
        # Extracting maxima coordinates
        # ----------------------------------------------------------------------------------------------------------
        xx = np.array(list(range(dog_space.shape[1])))
        yy = np.array(list(range(dog_space.shape[0])))
        zz = np.array(list(range(dog_space.shape[2])))
        xx, yy = np.meshgrid(xx, yy)
        xx = np.tile(xx[..., np.newaxis], [1, 1, dog_space.shape[2]])
        yy = np.tile(yy[..., np.newaxis], [1, 1, dog_space.shape[2]])
        zz = np.tile(zz[np.newaxis, np.newaxis, ...], [dog_space.shape[1], dog_space.shape[0], 1])

        xx_max = xx[max_3d]
        yy_max = yy[max_3d]
        zz_max = zz[max_3d]
        values = dog_space[yy_max, xx_max, zz_max]

        return np.hstack((xx_max[..., np.newaxis], yy_max[..., np.newaxis], zz_max[..., np.newaxis], values[..., np.newaxis]))
