# instead of scikit-image you could use matplotlib and opencv to read, write, and display images
import numpy as np
import skimage as sk
import skimage.io as skio

# name of the input file
imname = 'proj1/images/cathedral.jpg'

# read in the image
im = skio.imread(imname)

# convert to double (might want to do this later on to save memory)    
im = sk.img_as_float(im)
    
# compute the height of each part (just 1/3 of total)
height = int(np.floor(im.shape[0] / 3.0))

# separate color channels
b = im[:height]
g = im[height: 2*height]
r = im[2*height: 3*height]

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

def align(moving, ref):
    # TODO: search over displacements, score each with a metric (L2 / NCC),
    # and return np.roll(moving, best_shift, axis=(0, 1)).
    # placeholder: no alignment
    return moving

ag = align(g, b)
ar = align(r, b)
# create a color image
im_out = np.dstack([ar, ag, b])

# save the image
fname = 'proj1/out/cathedral.jpg'
skio.imsave(fname, sk.img_as_ubyte(im_out))

# display the image
skio.imshow(im_out)
skio.show()
