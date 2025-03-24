import os
import random
import numpy as np
from calendar import c
import cv2 as cv
import numpy as np
import os
import time
from tqdm import tqdm

import os
import random
import numpy as np
import cv2 as cv


def edof_sim(
    img: np.ndarray,
    depth: int,
    object_z_pos: int,
    dov: int,
    blur_multiplier: int,
    noise_strength: int,
):
    """Takes an image and returns an edof simulation of the given image. Uses a blur function that is linearly
    dependent on z distance. For the simulation a simulated volume with the given depth will be created in which
    the given image is at the given z-position. The "focal plane" is then moved through this volume and at every
    step an image is captured. The result is the mean of all captured images.
    Args:
        img (np.ndarray): gray scale image on which the simulation should be applied
        depth (int): the depth of the simulated volume. A bigger depth results in a more blurred image
        object_z_pos (int): the z position of the object in the simulated volume
        dov (int): depth of view, i.e. the z distance the focal plane spans.
        blur_multiplier (int): regaluates the strength of the blur function
        noise_strength (int): strength of noise. Should be between 0 and 255

    Returns:
        _type_: _description_
    """
    #s = time.perf_counter()
    res = np.zeros(img.shape, dtype=np.float64)
    counter = 0
    # move focal plane
    # the z-position of the focal plane is given by
    # z(t)= depth / 2 * (1 - cos(w*t))
    # (solve the differential equation with
    # dz/dt = sin(t) with start-values z(0)=0 and z(2*pi/w)=depth)
    # assume w=1
    for t in np.arange(0, 2 * np.pi, 0.1):
        counter += 1
        f_z = depth / 2 * (1 - np.cos(t))

        z_dist_object = abs(f_z - object_z_pos)
        noise = np.random.sample(img.shape) * noise_strength
        if z_dist_object > dov:
            # object is not in focal plane and thus blurred
            blurred = cv.blur(
                img,
                (
                    blur_multiplier * round(z_dist_object),
                    blur_multiplier * round(z_dist_object),
                ),
            )
            res += blurred
        else:
            # object is in focal plane and thus sharp
            res += img.astype(np.float64)
        res += noise
        # prevent overflow from noise
        res[np.where(res > 255 * counter)] = 255 * counter

    res /= counter
    res = res.astype(np.uint8)
    #print(time.perf_counter() - s)
    return res

def add_granular_noise(blurred_img, noise_strength):
    # Add granular noise to the blurred image
    noise = np.random.normal(loc=0.9, scale=noise_strength, size=blurred_img.shape)
    noisy_blurred_img = np.clip(blurred_img * noise, 0, 255).astype(np.uint8)
    noisy_blurred_img = cv.blur(noisy_blurred_img+255, (40,40))
    #noisy_blurred_img = cv.medianBlur(noisy_blurred_img, 7)

    return noisy_blurred_img

def main(source_path, output_path, num_images, num_artifacts, index):
    # Get a list of all subfolders in the source path
    subfolders = [f for f in os.listdir(source_path) if os.path.isdir(os.path.join(source_path, f))]
    # Define the list of obligatory subfolders
    obligatory_subfolders = ["artifacts", "detritus_blob", "detritus_filamentous", "diatom_chain_string"]

    # Create an initial white canvas of 2560x2560 pixels (grayscale)
    canvas = np.ones((2560, 2560), dtype=np.uint8) * 255

    # Create a copy of the canvas to store the ground truth
    ground_truth = canvas.copy()

    canvas = add_granular_noise(canvas, noise_strength=0.1)
    #cv.imwrite('/home/plankton/Data/edof_sim/noise_test.png', blurred_canvas)

    for _ in range(num_images):
        # Randomly select a subfolder
        selected_subfolder = random.choice(subfolders)
        subfolder_path = os.path.join(source_path, selected_subfolder)

        # Get a list of image filenames in the selected subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly select an image from the list
        selected_image = random.choice(image_files)

        # Load the image using OpenCV (grayscale)
        img_path = os.path.join(subfolder_path, selected_image)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        threshold = 200
        object = img < threshold
        img = np.where(object, img, 255)
    

        # Calculate the padding size as 20% of the original image size
        pad_amount = int(max(img.shape) * 0.5)

        # Pad the image
        padded_img = np.pad(img, ((pad_amount, pad_amount), (pad_amount, pad_amount)), mode='constant', constant_values=255)
        img = padded_img

        # Generate a random object_z_pos between 0 and 50
        object_z_pos = random.uniform(0, 50)

        # Call the edof_sim function
        result_img = edof_sim(img, depth=50, object_z_pos=object_z_pos, dov=1, blur_multiplier=4, noise_strength=30)

        # Generate random positions to place the result_img on the canvas
        x_pos = random.randint(0, canvas.shape[1] - result_img.shape[1])
        y_pos = random.randint(0, canvas.shape[0] - result_img.shape[0])
        # Add the result_img pixel values to the canvas without overflow
        x_end = x_pos + result_img.shape[1]
        y_end = y_pos + result_img.shape[0]
        canvas[y_pos:y_end, x_pos:x_end] = np.minimum(canvas[y_pos:y_end, x_pos:x_end] + result_img, 255)

        # Paste the source image onto the ground truth canvas
        ground_truth[y_pos:y_end, x_pos:x_end] = np.minimum(ground_truth[y_pos:y_end, x_pos:x_end] + img, 255)

    for _ in range(num_artifacts):
        # obligatry subfolders
        selected_subfolder = random.choice(obligatory_subfolders)
        subfolder_path = os.path.join(source_path, selected_subfolder)

        # Get a list of image filenames in the selected subfolder
        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Randomly select an image from the list
        selected_image = random.choice(image_files)

        # Load the image using OpenCV (grayscale)
        img_path = os.path.join(subfolder_path, selected_image)
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

        # Calculate the padding size as 20% of the original image size
        pad_amount = int(max(img.shape)*0.75)

        # Pad the image
        padded_img = np.pad(img, ((pad_amount, pad_amount), (pad_amount, pad_amount)), mode='constant', constant_values=255)
        img = padded_img

        # Generate a random object_z_pos between 0 and 50
        object_z_pos = random.uniform(0, 50)

        # Call the edof_sim function
        result_img = edof_sim(img, depth=50, object_z_pos=object_z_pos, dov=1, blur_multiplier=5, noise_strength=10)

        # Generate random positions to place the result_img on the canvas
        x_pos = random.randint(0, canvas.shape[1] - result_img.shape[1])
        y_pos = random.randint(0, canvas.shape[0] - result_img.shape[0])
        # Add the result_img pixel values to the canvas without overflow
        x_end = x_pos + result_img.shape[1]
        y_end = y_pos + result_img.shape[0]
        canvas[y_pos:y_end, x_pos:x_end] = np.minimum(canvas[y_pos:y_end, x_pos:x_end] + result_img, 255)

        # Paste the source image onto the ground truth canvas
        ground_truth[y_pos:y_end, x_pos:x_end] = np.minimum(ground_truth[y_pos:y_end, x_pos:x_end] + img, 255)

    # Normalize the canvas and ground truth to 255
    canvas = (canvas * (255 / canvas.max())).astype(np.uint8)
    ground_truth = (ground_truth * (255 / ground_truth.max())).astype(np.uint8)

    # Save the final canvas as an image (grayscale)
    output_image_path = os.path.join(output_path, "blurred/%d.png"%index)
    cv.imwrite(output_image_path, canvas)

    # Save the ground truth image
    ground_truth_path = os.path.join(output_path, "gt/%d.png"%index)
    cv.imwrite(ground_truth_path, ground_truth)

if __name__ == "__main__":
    source_path = "/home/plankton/Data/PlanktonSet/train"
    output_path = "/home/plankton/Data/edof_sim"
    num_images = 20  # This is the number of plankton objects on the generated image
    num_artifacts = 200 #number of artifacts (including phytoplankton)
    im_num = 1500   #number of images that will be generated in the output_path
    index = 0 #starting index
    for _ in tqdm(range(im_num)):
        main(source_path, output_path, num_images, num_artifacts, index)
        index += 1
