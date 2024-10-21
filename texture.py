import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# Function to resize images 
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size)

# Reading an image and converting it into grayscale 
image = cv2.imread("stone2.jpg")
image_resized = resize_image(image)
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Finding texture using Sobel operator for reference image
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_texture = cv2.magnitude(sobel_x, sobel_y)

# Creating a list to store images 
classes = {
    'Water Texture': [
        cv2.imread("water1.jpeg"),
        cv2.imread("water2.jpeg"),
        cv2.imread("water3.jpeg")
    ],
    'Sun Texture': [
        cv2.imread("sun1.jpeg"),
        cv2.imread("sun2.jpeg"),
        cv2.imread("sun3.jpeg")
    ],
    'Stone Texture': [
        cv2.imread("stone1.jpg"),
        cv2.imread("stone2.jpg"),
        cv2.imread("stone3.jpg")
    ]
}

# Check images loaded are not
if not any(classes.values()):
    print("No images loaded in any class.")
else:
    # Selecting a random image 
    random_label, random_images = random.choice(list(classes.items()))
    random_image = random.choice(random_images)

    # Resize the random image
    random_image_resized = resize_image(random_image)
    gray_random = cv2.cvtColor(random_image_resized, cv2.COLOR_BGR2GRAY)

    # Finding texture of random image
    sobel_x_random = cv2.Sobel(gray_random, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y_random = cv2.Sobel(gray_random, cv2.CV_64F, 0, 1, ksize=5)
    random_texture = cv2.magnitude(sobel_x_random, sobel_y_random)

    # calculate Euclidean distance
    def calculate_distance(texture1, texture2):
        return np.linalg.norm(texture1 - texture2)

    # Defining label for reference image
    reference_label = "Stone Texture"

    # Checking if the random image is from the same class
    if random_label == reference_label:
        print("The images belong to the same class.")
    else:
        print("The images belong to different classes.")

    # Calculate distance between the reference and random image textures
    distance = calculate_distance(sobel_texture.flatten(), random_texture.flatten())
    print(f"The Euclidean distance between the textures is: {distance:.2f}")

    # Displaying the images 
    plt.subplot(1, 2, 1)
    plt.imshow(random_texture, cmap='gray')
    plt.title(f"Random Image\nClass: {random_label}")

    plt.subplot(1, 2, 2)
    plt.imshow(sobel_texture, cmap='gray')
    plt.title(f"Reference Image\nClass: {reference_label}")

    plt.suptitle(f"Euclidean Distance: {distance:.2f}")
    plt.show()
