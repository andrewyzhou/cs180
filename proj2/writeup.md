# part 1: fun with filters

## 1.1 convolutions from scratch

convolution slides a kernel over the image and, at every pixel, multiplies the kernel with the patch of image under it and sums the result. before sliding, the kernel is flipped both top to bottom and left to right. without the flip the operation is correlation, which gives the same result for symmetric kernels like the box filter but flips the sign of the difference operators. the flip is also what makes convolution associative, which part 1.3 relies on.

i implemented it two ways with numpy only. the four loop version loops over every output pixel (two loops) and, for each one, over every entry of the kernel (two more loops). the two loop version keeps the loops over output pixels and replaces the kernel loops with one numpy multiply and sum over the patch.

```python
def zero_pad(img, kernel):
    # same padding, extra row/col on top/left for even kernels like scipy
    kh, kw = kernel.shape
    top, left = kh // 2, kw // 2
    bottom, right = kh - 1 - top, kw - 1 - left
    padded_img = numpy.zeros((img.shape[0] + top + bottom, img.shape[1] + left + right))
    padded_img[top:top + img.shape[0], left:left + img.shape[1]] = img
    return padded_img

# four for loops
def conv_4loops(img, kernel):
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    padded = zero_pad(img, kernel)
    out = numpy.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            total = 0.0
            for m in range(kh):
                for n in range(kw):
                    total += k[m, n] * padded[i + m, j + n]
            out[i, j] = total
    return out

# two for loops
def conv_2loops(img, kernel):
    k = kernel[::-1, ::-1]
    kh, kw = k.shape
    padded = zero_pad(img, kernel)
    out = numpy.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = numpy.sum(padded[i:i + kh, j:j + kw] * k)
    return out
```

both use "same" padding with zeros, so the output is the same size as the input. for a kernel with an even number of rows or columns, like D_x (1x2), there is no exact center, so the extra row or column of padding goes on the top or left. this matches what `scipy.signal.convolve2d` does with `mode="same"`.

i ran both on a grayscale picture of myself (512x512) with a 9x9 box filter (every entry 1/81), D_x = [1, -1], and D_y, which is D_x transposed. both versions match `convolve2d` on all three kernels.

| kernel | four loops | two loops | convolve2d |
|---|---:|---:|---:|
| 9x9 box | 3.95s | 0.46s | 0.024s |
| D_x | 0.15s | 0.45s | 0.002s |
| D_y | 0.16s | 0.45s | 0.003s |

*todo: figure of the original, box blur, D_x, D_y*

**runtime.** the two loop version is about 8x faster than four loops on the box filter, since numpy does the 81 multiplies and adds at once instead of python doing them one at a time. on D_x and D_y it is actually slower, because those kernels only have 2 entries and building a numpy patch for every pixel costs more than 2 python multiplies. convolve2d is written in c and is 20x faster than my fastest version on the box filter and over 60x faster on the difference operators.

**boundaries.** at the edge of the image part of the kernel hangs off the image, so the padding decides what values it sees there. zero padding treats everything outside the image as black. on the box filter, edge pixels get averaged with up to half zeros, which leaves a dark frame about 4 pixels thick around the blurred image. on D_x and D_y, the jump from the image to the black padding shows up as a fake edge along the whole border. convolve2d also pads with zeros by default (`boundary="fill"`), which is why the results match exactly, but it can also mirror the image at the border (`boundary="symm"`) or wrap around to the other side (`boundary="wrap"`). mirroring removes both the dark frame and the fake edges, so i use it for the rest of the project.

*todo: corner crop of the box blur showing the dark frame, zero padding vs symm*

## 1.2 finite difference operator

convolving the cameraman image with D_x gives each pixel minus the pixel to its left, so it is near 0 in flat areas and large where brightness changes from left to right, which picks out vertical edges. D_y does the same from top to bottom and picks out horizontal edges. both can be negative (light to dark instead of dark to light), so in the images below gray is 0, white is positive, and black is negative. from here on i use `convolve2d` with `boundary="symm"` so the image border does not register as an edge.

```python
dx = convolve2d(cam, Dx, mode="same", boundary="symm")
dy = convolve2d(cam, Dy, mode="same", boundary="symm")
mag = numpy.sqrt(dx ** 2 + dy ** 2)
edges = mag > 0.25
```

the gradient magnitude combines the two into one number for how strong the edge is at each pixel. treating (dx, dy) as a vector, the magnitude is its length, sqrt(dx² + dy²). this also removes the sign, so both directions of an edge come out positive and an edge at an angle, which shows up partly in dx and partly in dy, still gets its full strength.

*todo: figure of dx, dy, gradient magnitude (out/1_2_dx.png, out/1_2_dy.png, out/1_2_mag.png)*

to turn the magnitude into an edge image, i binarized it, marking every pixel above a threshold as an edge. i tried thresholds from 0.1 to 0.35. at 0.1 and 0.15 every real edge shows up, but so does the grass, which is full of small brightness changes and turns into speckle. at 0.3 and above the speckle is gone, but the outline of the coat and the tripod legs start breaking apart. i went with 0.25, which keeps the main outlines mostly connected and leaves only a little noise in the grass. the rectangle around the border is not from padding, since the cameraman file has a thin white border built into the image and the step from it into the photo is a real edge.

*todo: threshold sweep (out/1_2_threshold_sweep.png) and final edge image (out/1_2_edges.png)*

## 1.3 derivative of gaussian (dog) filter

the edges from 1.2 pick up a lot of noise from the grass, so here i blur the image first with a gaussian. i made the 2d gaussian by taking the 1d kernel from `cv2.getGaussianKernel(13, 2)` (sigma 2, 13 entries, enough to cover 3 sigma on each side) and multiplying it with its transpose to get a 13x13 kernel. it is a weighted average that counts nearby pixels the most, so it smooths out small brightness changes like grass texture while keeping the large ones.

```python
g = cv2.getGaussianKernel(13, 2)
G = g @ g.T
blur = convolve2d(cam, G, mode="same", boundary="symm")
bdx = convolve2d(blur, Dx, mode="same", boundary="symm")
bdy = convolve2d(blur, Dy, mode="same", boundary="symm")
bmag = numpy.sqrt(bdx ** 2 + bdy ** 2)
```

*todo: figure of blurred, dx, dy, gradient magnitude (out/1_3_blur.png, out/1_3_dx.png, out/1_3_dy.png, out/1_3_mag.png)*

after the blur, almost all of the grass speckle is gone and the outlines of the coat and tripod are continuous instead of broken up. the edges are thicker, since the blur spreads each edge over a few pixels. the magnitudes are also smaller overall, because blurring makes every brightness jump less steep, so the threshold had to come down from 0.25 to 0.06. i picked 0.06 from a sweep of 0.04 to 0.12: lower kept some noise along the bottom, higher started breaking the outline of the coat.

*todo: threshold sweep (out/1_3_threshold_sweep.png) and finite difference vs blurred edges side by side (out/1_2_edges.png, out/1_3_edges.png)*

the same result can be done with one convolution instead of two. since convolution is associative, blurring then taking the difference, (image * G) * D_x, is the same as image * (G * D_x). so i convolved G with D_x and D_y once to get two derivative of gaussian filters, then applied each to the image directly. each one looks like a blurred version of [1, -1], a positive blob next to a negative blob.

*todo: figure of G and the two dog filters (out/1_3_filters.png)*

```python
DoGx = convolve2d(G, Dx)
DoGy = convolve2d(G, Dy)
ddx = convolve2d(cam, DoGx, mode="same", boundary="symm")
ddy = convolve2d(cam, DoGy, mode="same", boundary="symm")
```

the two methods match. away from the border, the largest difference in dx is 9.5e-16 and in dy 8.3e-16, which is just floating point rounding, and the edge images are the same pixel for pixel. right at the border they differ slightly, because mirroring the image before the blur is not the same as mirroring the blurred image.

*todo: blur then difference vs dog edges side by side (out/1_3_edges.png, out/1_3_dog_edges.png)*

# part 2: fun with frequencies

for part 2 i switched the blurring to `cv2.sepFilter2D` and `cv2.filter2D` with mirrored borders, since my loop convolutions are too slow on photos this size. on the taj image the cv2 version matches `convolve2d` with `boundary="symm"` to within 2.4e-14. i also shrank the efros and ng photos to 2600 px tall (about 600 kb each) so the notebook runs in under a minute.

## 2.1 image "sharpening"

a gaussian blur keeps the low frequencies of an image and removes the high ones. subtracting the blurred image from the original leaves only the high frequencies, which are the fine detail and sharp edges. adding some of those back on top of the original makes the image look sharper:

sharpened = f + alpha * (f - f * G)

since convolution is linear, this collapses into one kernel. writing e for the impulse (all zeros with a 1 in the center, so f * e = f), the sharpened image is f * ((1 + alpha) e - alpha G), and that kernel is the unsharp mask filter.

```python
def unsharp_kernel(sigma, alpha):
    ksize = 2 * int(numpy.ceil(3 * sigma)) + 1
    g = cv2.getGaussianKernel(ksize, sigma)
    impulse = numpy.zeros((ksize, ksize))
    impulse[ksize // 2, ksize // 2] = 1
    return (1 + alpha) * impulse - alpha * (g @ g.T)
```

on the taj image with sigma 2, the single unsharp convolution matches doing it in steps (blur, subtract, add) to within 2.4e-14.

*todo: taj original, blurred, high frequencies, sharpened (out/2_1_taj_blur.png, out/2_1_taj_high.png, out/2_1_taj_alpha1.png)*

alpha controls how much of the high frequencies get added back. at 0.5 and 1 the carvings on the building and the edges of the trees get crisper without looking processed. at 2 the image starts looking oversharpened, and at 5 there are dark and light halos along every strong edge, like the outline of the dome against the sky, and the noise in the sky gets amplified into visible grain. the halos come from the high frequency image overshooting on both sides of an edge, and scaling that up by 5 exaggerates it.

*todo: taj alpha sweep (out/2_1_taj_alpha0.png through out/2_1_taj_alpha5.png)*

i also sharpened my own portrait with alpha 0, 1, and 3. alpha 1 is slightly rough, while alpha 3 makes my skin look real rough and we can see my pores and all that </3.

*todo: portrait alphas (out/2_1_andrew_alpha0.png, out/2_1_andrew_alpha1.png, out/2_1_andrew_alpha3.png)*

to check how much sharpening actually recovers, i blurred a sharp photo (ng) with sigma 3, then sharpened the blurred version with the same sigma. i measured how close each one is to the original with psnr (peak signal to noise ratio, higher means closer to the original).

| image | psnr |
|---|---:|
| blurred | 34.35 db |
| resharpened, alpha 0.5 | 34.87 db |
| resharpened, alpha 1 | 35.30 db |
| resharpened, alpha 2 | 35.74 db |
| resharpened, alpha 4 | 34.70 db |

*todo: eye crops (out/2_1_ng_eye_original.png, out/2_1_ng_eye_blur.png, out/2_1_ng_eye_alpha*.png)*

sharpening helps a little up to alpha 2 and then gets worse again. looking at the eyes, the resharpened versions have crisper outlines around the eyes and eyebrows, but the eyelashes and skin texture from the original never come back. the blur already removed those frequencies, and unsharp masking can only boost the frequencies that are still in the image, so it adds contrast to edges but cannot add back detail. past alpha 2 it mostly adds halos and grain, which is why the psnr drops.

## 2.2 hybrid images

a hybrid image adds the low frequencies of one image to the high frequencies of another. up close the high frequencies dominate and you see the second image, and from far away only the low frequencies are visible and you see the first one.

```python
def hybrid(im_low, im_high, s_low, s_high):
    low = blur(im_low, s_low)
    high = im_high - blur(im_high, s_high)
    return low, high, numpy.clip(low + high, 0, 1)
```

the two images need to be lined up first. the starter code aligns by clicking two points in each image, but i hardcoded the two eye positions for each photo instead so the notebook reruns without clicking. from the two pairs of points i fit a transform made of a rotation, a uniform scale, and a shift that moves one image's eyes onto the other's, then warp it into the other image's frame.

to pick the blur amounts i used the cutoff frequency, the frequency where the gaussian cuts the strength of the image in half. for a gaussian with standard deviation sigma pixels, the cutoff is about 0.187 / sigma cycles per pixel.

**derek and nutmeg**

derek is the low pass image and nutmeg the high pass one, since a cat's fur and whiskers are already mostly high frequency. i used sigma 10 for derek (cutoff 0.0187 cycles per pixel, so detail smaller than about 53 px is removed) and sigma 6 for nutmeg (cutoff 0.0312, so only detail smaller than about 32 px is kept). i did this one in grayscale.

*todo: inputs and aligned images (out/2_2_derek.png, out/2_2_nutmeg_aligned.png)*

*todo: low pass derek, high pass nutmeg, hybrid (out/2_2_derek_low.png, out/2_2_nutmeg_high.png, out/2_2_derek_nutmeg.png)*

the log magnitude of the fourier transform shows where each image's energy is, with low frequencies in the center and higher ones further out. the low passed derek only has a bright spot in the middle, the high passed nutmeg has a dark hole in the middle with the rest kept, and the hybrid has both. the bright horizontal and vertical lines through the center of every transform come from the image's edges, since the fourier transform treats the image as repeating and the left edge doesn't match the right one.

*todo: fft figure (out/2_2_fft.png)*

*todo: hybrid at shrinking sizes (out/2_2_derek_nutmeg_sizes.png)*

**color (bells and whistles)**

i tried four versions: both parts gray, color only in the low part, color only in the high part, and color in both. color in the low part works best. most of an image's color is in large smooth areas (derek's skin, the background), which are low frequency, so it strengthens the far away view. color in the high part barely shows, since the high pass image is close to zero everywhere except edges, so its color only shows up as faint tints on the whiskers. color in both looks almost the same as color in the low part only.

*todo: four color versions (out/2_2_color_gray.png, out/2_2_color_color_low.png, out/2_2_color_color_high.png, out/2_2_color_color_both.png)*

**me and dwight**

i'm the low pass image (sigma 10) and dwight is the high pass one (sigma 8), aligned into my photo's frame. up close you see dwight's glasses, eyebrows, and face, and from far away it turns back into me in a suit. it works well because dwight's glasses and the edges of his hood are strong high frequency features. the outline of the hood still shows faintly against my white background up close.

*todo: me, dwight, hybrid, sizes (out/2_2_dwight_aligned.png, out/2_2_me_dwight.png, out/2_2_me_dwight_sizes.png)*

**efros and ng**

i first tried ng as the low pass image and efros as the high pass one, but efros's glasses stayed visible at every distance, since thin dark frames are exactly the kind of feature a high pass keeps. swapping them worked better, so efros is the low pass image (sigma 35) and ng the high pass one (sigma 18), with efros aligned into ng's frame. up close you see ng's eyes and smile on efros's head, and from far away you see efros with his glasses. ng's photo also didn't reach the bottom of efros's frame after alignment, which left smeared stripes, so warping efros into ng's frame fixed that too.

*todo: efros, ng, hybrid, sizes (out/2_2_efros_aligned.png, out/2_2_efros_ng.png, out/2_2_efros_ng_sizes.png)*

## 2.3 gaussian and laplacian stacks

a gaussian stack is the image blurred more and more at each level, without shrinking it like a pyramid would. a pyramid halves the image at each level, so each level's blur covers twice as many of the original's pixels. to match that, i double sigma each level, starting at 2. a laplacian stack is the difference between neighboring levels of the gaussian stack, so each level holds one band of frequencies, from the finest detail at level 0 to the coarsest at the end. the last level is just the last gaussian level, so adding up the whole laplacian stack gives back the original image exactly (the largest error is 1.1e-16).

```python
def gaussian_stack(im, levels, sigma0):
    stack = [im]
    for i in range(1, levels):
        stack.append(blur(stack[-1], sigma0 * 2 ** (i - 1)))
    return stack

def laplacian_stack(im, levels, sigma0):
    g = gaussian_stack(im, levels, sigma0)
    return [g[i] - g[i + 1] for i in range(levels - 1)] + [g[-1]]
```

i used 6 levels on the apple and the orange. the laplacian levels are mostly close to zero, so for display i stretched each one to fill the full range from black to white.

*todo: apple and orange stacks (out/2_3_apple_stacks.png, out/2_3_orange_stacks.png)*

## 2.4 multiresolution blending

to blend two images, i build laplacian stacks for both and a gaussian stack for the mask, which is 1 where the first image should show and 0 where the second should. at every level, the blended band is mask times the first image's band plus (1 - mask) times the second's, and adding up all the blended bands gives the result. the mask gets blurrier at each level, so the finest detail is blended across a narrow seam and the coarse colors across a wide one. that way there is no ghosting of fine detail and no visible step in color.

```python
def blend(a, b, mask, levels, sigma0):
    la = laplacian_stack(a, levels, sigma0)
    lb = laplacian_stack(b, levels, sigma0)
    gm = [m[..., None] for m in gaussian_stack(mask, levels, sigma0)]
    parts = [(m * x, (1 - m) * y) for m, x, y in zip(gm, la, lb)]
    return numpy.clip(sum(x + y for x, y in parts), 0, 1), parts
```

**oraple**

the mask is the left half of the image. without blending there is a hard vertical line down the middle, and with it the apple fades into the orange.

*todo: apple, orange, no blending, oraple (out/2_4_oraple.png)*

below is my version of figure 3.42 from szeliski. the first three rows are laplacian levels 0, 2, and 4, showing the masked apple, the masked orange, and their sum. the last row adds up every level. at level 0 the seam is sharp, and by level 4 it is spread over a wide band.

*todo: band figure (out/2_4_oraple_bands.png)*

**half efros, half ng**

the same vertical seam, placed between the eyes of the aligned photos, with 7 levels starting at sigma 3. without blending, the line between efros's lighter skin and hair and ng's darker hair is obvious. after blending, the skin tones meet smoothly and the main remaining giveaway is the hair, which changes color over a wide band across the forehead.

*todo: efros aligned, ng, no blending, blended (out/2_4_efros_ng.png)*

**my face in dwight's hood (irregular mask)**

i aligned my portrait into the dwight frame by the eyes and used an oval mask around dwight's face, with 6 levels starting at sigma 2. without blending, the oval's edge is a hard line and my skin is a different color from dwight's. after blending, the edge disappears and the colors fade into each other. dwight's double chin is visible under mine.

*todo: dwight, me aligned, mask, no blending, blended (out/2_4_me_hood.png)*

*todo: band figure for this blend, similar to figure 10 in the paper (out/2_4_me_hood_bands.png)*

**color (bells and whistles)**

every blend here is done in color, with each color channel blended separately using the same mask. in color, the fade between the two skin tones and between efros's and ng's hair colors is the most visible part of the blend, and the multiresolution blend handles it by spreading the change in color over the widest band.

# most important thing i learned

the most important thing i learned was the wealth of features and data available in images that i don't really think about, which was illuminated by doing these transformations and kernels. it made me appreciate my human brain for learning to process these things so effortlessly and efficiently, and it makes me more interested in ml and the brain and fascinated with evolution.
