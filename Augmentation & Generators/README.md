# Augmentation

Improving image classification through image augmentation is a well known technique. <sup>1</sup> 

The NN model will reach an apogee if for every epoch the same amount of images are seen. Instead rotation, cropping, or flipping can add more unique instances. Therefore, within each epoch the same amount of images are seen (time constraint) but the number of unique images increases with each epoch.

![freeze_NN](https://user-images.githubusercontent.com/57273222/95635094-808ca480-0a59-11eb-8e2b-df3b52459839.PNG)

As expected, after our training (fit method) the weights are different.

![freeze2](https://user-images.githubusercontent.com/57273222/95683725-1b50c480-0bbb-11eb-86aa-d6eb7742dfce.PNG)

We can refer which layer to freeze using: model.layers[" "].trainable = False
In the case of the first layer.

![freeze3](https://user-images.githubusercontent.com/57273222/95683817-97e3a300-0bbb-11eb-9df1-40885096ad9b.PNG)

## Transfer Learning

The real benefit surfaces once we compare the test accuracies of a training with and w/o transfer learning. 
We also have to freeze the transfered model to avoid tremendous amount of training time. As the transfered model is already optimized.

Combining the famous [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) and [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) we will demonstrate the transfer learning advantage.

Using the functional API of Tensorflow we create a function to remove the output layer of the pre trained model. We add an extra layer followed by Dropout with the Sequential API this time. We end with a single unit final output layer. 

The final result shown on a table and Confusion Matrix.

## Sources
1. Perez, L., & Wang, J. (2017). The Effectiveness of Data Augmentation in Image Classification using Deep Learning. ArXiv, abs/1712.04621.
