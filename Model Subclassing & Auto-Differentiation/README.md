## Model Sub-classing

Tensorflow offers many methods for building models such as the *sequential* or *functional* API. For even more control, a lower level exists called the *model sub-classing* API. It gives more control over both the construction and operation of the deep neural network.

Similarly to the *functional* API, we inherit from the model class (*super*). We create layers of the NN as a layer object and assign it as a class attribute in the initializer (or also known as the constructor). The forward pass is defined in thecall method, which in turn will build the layers defined in the constructor. The same call method also takes a *training* keyword argument which tells the model if you are training or testing (validating). Finally, the dropout layer is activated using the training Boolean variable.

```
class MyModel(Model):

  def__init__(self, num_classes, **kwargs):
    super(MyModel, self).__init__(**kwargs)
    self.dense1 = Dense(16, activation='sigmoid')
    self.dropout = Dropout(0.5)
    self.dense2 = Dense(num_classes, activation='softmax')
    
   def call(self, inputs, training=False):
    h = self.dense1(inputs)
    h = self.dropout(h, training=training)
    return self.dense2(h)
    
my_model = MyModel(12, name='my_model')
```

## Auto-Differentiation

Up until now, we relied on the model.fit method to train our NN. For more custom algorithm training loops we can use the sub-classing API combined with the auto-differentation capability of tensorflow to provide more control over the model. Calling tape.watch on the tensor x will start recording on the variable x as the independent variable. Here is an example:

```
import tensorflow as tf

x = tf.constant([-1, 0, 1], dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.math.exp(x)
    z = 2 * tf.reduce_sum(y)
    dz_dx = tape.gradient(z, x)
```
![CodeCogsEqn (1)](https://user-images.githubusercontent.com/57273222/98587786-a1cef380-2298-11eb-9f67-d0e54af22f5e.gif)

![CodeCogsEqn (2)](https://user-images.githubusercontent.com/57273222/98587948-e65a8f00-2298-11eb-9dd0-b92b7a56bc6d.gif)



