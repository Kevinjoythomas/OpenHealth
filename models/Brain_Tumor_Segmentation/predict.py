import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow,imread



def highlight_tumor(image_path):
    model_path = 'brain_tumor_segmentation.hdf5'
    model = load_model(model_path, compile=False)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0

    image = np.expand_dims(image, axis=(0, -1))

    mask = model.predict(image)

    threshold = 0.5
    mask_binary = (mask > threshold).astype(np.uint8)

    original_image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    highlight_mask_resized = cv2.resize(mask_binary[0], (original_image_rgb.shape[1], original_image_rgb.shape[0]))

    highlight_mask = cv2.cvtColor(highlight_mask_resized, cv2.COLOR_GRAY2RGB)
    highlight_mask[:, :, 0] = np.where(highlight_mask[:, :, 0] > 0, 255, 0)  
    highlight_mask[:, :, 1] = 0  
    highlight_mask[:, :, 2] = 0 

    highlighted_image = cv2.addWeighted(original_image_rgb, 0.7, highlight_mask, 0.3, 0)

    plt.imshow(highlighted_image)
    plt.title('Highlighted Tumor')
    plt.axis('off')
    plt.show()

    
image_path = 'test.png'
highlight_tumor(image_path)