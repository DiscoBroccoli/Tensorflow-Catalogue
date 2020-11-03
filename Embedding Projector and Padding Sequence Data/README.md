## Embedding Projector

Embedding layers are one of the most essential tools for natural language processing or time-series forecasting. More practically, they transform a word into a vector. 
In turn, these are trained in a supervised task.

The outcome will give the neural network the ability to associate words like "good" and "best". 
We can visualize this using the embedding projector in the tensforflow framework. For this task, we use the IMDB dataset. 

The imdb dataset is initially formatted with the word index. Therefore, we simply need to swap the keys and values of the word index.
```
inv_imdb_word_index = {value: key for key, value in imdb_word_index.items()}
```

Using the functional API we build a the embedding layer.

```
review_sequence = tf.keras.Input((None, ))
embedding_sequence = tf.keras.layers.Embedding(input_dim=max_index_value+1, output_dim=embedding_dim)(review_sequence)                                               
average_embedding = tf.keras.layers.GlobalAveragePooling1D()(embedding_sequence)
positive_probability = tf.keras.layers.Dense(units=1, activation='sigmoid')(average_embedding)

model = tf.keras.Model(inputs=review_sequence, outputs=positive_probability)
```

After saving the embedding matrix, and labels we can import the files (meta.tsv, and vecs.tsv) in https://projector.tensorflow.org/

![imdb_sentiment](https://user-images.githubusercontent.com/57273222/98024034-bd885480-1dd5-11eb-8105-9fd7b499666f.png)

![list2](https://user-images.githubusercontent.com/57273222/98024710-b1e95d80-1dd6-11eb-8384-1ec380685f49.png)

We notice a clear separation for words that belongs in either "good" or "bad" reviews. Additionally, words that are synonymous have similar embedding paramters 
(closer to each other).

## Padding Sequence Data
A small caveate to be addressed is the sequence lenght. In fact, not all batch entries will ave the same sentence lenght. 
Therefore a padding layer was applied during the preprocessing. The padding essentially fills up the empty spaces with the default value "0". 
Specifying a maximum length will truncate the extra letters. 

```
def make_padded_dataset(sequence_chunks):
    """
    This function takes a list of lists of tokenized sequences, and transforms
    them into a 2D numpy array, padding the sequences as necessary according to
    the above specification. The function should then return the numpy array.
    """
    
    padded_sequence_chunks = tf.keras.preprocessing.sequence.pad_sequences(sequence_chunks, 
                                                                           maxlen=500, 
                                                                           truncating='pre', 
                                                                           padding='pre', value=0)
    return padded_sequence_chunks
```

