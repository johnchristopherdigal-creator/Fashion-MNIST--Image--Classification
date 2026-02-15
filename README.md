# Fashion-MNIST--Image--Classification
google colab Link: https:https://colab.research.google.com/drive/1ZJsfqWMVSDULN-jwcMHUQF-eZKFpswnm?usp=drive_link
# 1.What is  the Fashion MNIST data set?
answer: Fashion MNIST is a collection of small black-and-white pictures of clothes.
Each picture shows one item, like a shirt, shoe, bag, or dress. The pictures are very small (28×28 pixels).
People use it to teach computers how to recognize different types of clothing. By showing the computer many examples, it slowly learns how to tell one item from another.
That’s it it’s just practice pictures to help a computer learn about clothes.
# 2.Why do we normalize image pixel values  before training?
We normalize pixel values by dividing them by 255 to scale them from 0–255 down to 0–1. This makes neural network training more stable and efficient, helping the model learn patterns faster and with fewer errors.
# 3.List the layers used  in the neural network and their function?
amswer:
layers.flatten and layers.dense 

from tensorflow.keras import layers
model = keras.Sequential([
layers.Flatten(input_shape=(28, 28)),
layers.Dense(128, activation='relu'),
layers.Dense(10)
])

# 4.what does an epoch mean in model training?
An epoch is one full pass of the training data through the network. Since the model can’t learn everything at once, we repeat this process multiple times—like a student reading a textbook over and over. Each pass helps the model catch mistakes, notice details it missed before, and gradually get better at recognizing different fashion items.

history = model.fit(train_images, train_labels, epochs=10)
output:
Epoch 1/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 9s 4ms/step - accuracy: 0.5636 - loss: 1.4883
Epoch 2/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 11s 5ms/step - accuracy: 0.7632 - loss: 0.6605
Epoch 3/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.7937 - loss: 0.5748
Epoch 4/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.8130 - loss: 0.5243
Epoch 5/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 10s 4ms/step - accuracy: 0.8223 - loss: 0.4990
Epoch 6/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.8351 - loss: 0.4716
Epoch 7/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.8394 - loss: 0.4605
Epoch 8/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 7s 4ms/step - accuracy: 0.8439 - loss: 0.4425
Epoch 9/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.8490 - loss: 0.4311
Epoch 10/10
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 8s 4ms/step - accuracy: 0.8514 - loss: 0.4252
# 5.compare the predicted label and actual label for the first test image?
Actual Label: It’s an Ankle boot (category 9).
Predicted Label: The model also picked Ankle boot.
Since they match, the model got it right! In your visualization, the label shows up in blue, which just means the prediction was correct.
step 2.6
import numpy as np
probability_model = keras.Sequential([
model,
layers.Softmax()
])
predictions = probability_model.predict(test_images)
print("Predicted label for first image:", np.argmax(predictions[0]))
print("Actual label:", test_labels[0])
output:
313/313 ━━━━━━━━━━━━━━━━━━━━ 1s 2ms/step
Predicted label for first image: 9
Actual label: 9
step 1.4
import matplotlib.pyplot as plt
class_names = [
'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(train_images[i], cmap='gray')
    plt.title(class_names[train_labels[i]])
    plt.axis('off')
plt.show()
# 6.What could we done top improven the model's accuracy?
To make your model better, the easiest win is switching to a CNN, since it’s made for understanding images. You can also help it learn more by adding extra layers, tweaking the training process, and mixing up the images so it can recognize clothes from different angles.
