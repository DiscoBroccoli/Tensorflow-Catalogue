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

![freeze_NN](https://user-images.githubusercontent.com/57273222/95635094-808ca480-0a59-11eb-8e2b-df3b52459839.PNG)

As expected, after our training (fit method) the weights are different.

![freeze2](https://user-images.githubusercontent.com/57273222/95683725-1b50c480-0bbb-11eb-86aa-d6eb7742dfce.PNG)

We can refer which layer to freeze using: model.layers[" "].trainable = False
In the case of the first layer.

![freeze3](https://user-images.githubusercontent.com/57273222/95683817-97e3a300-0bbb-11eb-9df1-40885096ad9b.PNG)


## Sources
1. Perez, L., & Wang, J. (2017). The Effectiveness of Data Augmentation in Image Classification using Deep Learning. ArXiv, abs/1712.04621.
