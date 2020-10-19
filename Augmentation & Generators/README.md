## Generators

Deep Neural Network (NN) commonly requires tremendous amount of data to perform well. Storing all the images into an array might require more memory than our computer offers. 

On the other hand, using dataset generators. We can load image one by one when we need it. Accelerating our code to require only enough memory to store a single image. 

```
def get_generator(features, labels, batch_size=1):
    for n in range(int(len(features)/batch_size)):
        yield (features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])
```
From the previous code extract, we can control the batch size with batch_size. In turn, this will change the amount of data that goes into our features and labels. 

```
train_generator = get_generator(training_features, training_labels, batch_size=10)
next(train_generator)
```
Notice, the generator will run out of samples if we run next() too many times. Exceeding the total number of samples.
*number of times next * batch_size > total samples available*. We can overcome this by adding a while true statement.

```
def get_generator_cyclic(features, labels, batch_size=1):
    while True:
        for n in range(int(len(features)/batch_size)):
            yield (features[n*batch_size: (n+1)*batch_size], labels[n*batch_size: (n+1)*batch_size])
        permuted = np.random.permutation(len(features))
        features = features[permuted]
        labels = labels[permuted]
```

To provide more context, a generator in Python is a function that returns an object that you can iterate over, but it doesn't store all those value in memory. Instead, it saves its own internal state, and each time we iterate the generator, it yields the next value in the series.

## Augmentation

Improving image classification through image augmentation is a well known technique. <sup>1</sup> 

For instance, the NN model will reach an accuracy apogee if for every epoch the same amount of images are seen.

Instead rotation, cropping, or flipping can add more unique instances. Therefore, within each epoch the same amount of images are seen (to keep same training time) but the number of unique images increases with each epoch. Essentially, improving the generalization of the model.

For this end, the [LSUN dataset](https://www.yf.io/p/lsun) is used.

![augment_data1](https://user-images.githubusercontent.com/57273222/96515137-c81de800-1232-11eb-939e-2121cd053813.PNG)

The augmented data has [rescaling](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) (because for typical learning rate it is easier to learn if the target values are between 0 and 1) rotation, brightness altered, and horizontal flip. 

Finally, the result between no-augmentation and augmentation is presented.
### No augmentation
![augment2](https://user-images.githubusercontent.com/57273222/96517259-a161b080-1236-11eb-9f6d-dd8e46563f00.PNG)
### With augmentation
![augment3](https://user-images.githubusercontent.com/57273222/96517280-aa528200-1236-11eb-8210-38093e1fa0a3.PNG)

## Sources
1. Perez, L., & Wang, J. (2017). The Effectiveness of Data Augmentation in Image Classification using Deep Learning. ArXiv, abs/1712.04621.
