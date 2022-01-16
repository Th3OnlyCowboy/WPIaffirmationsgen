import tensorflow as tf
import numpy as np
import os

## Much of tensorflow borrowed from Tensorflow tutorials and freecodecamp.org

# open affirmations
affirmations = open("Affirmations.txt", 'rb').read().decode(encoding='utf-8')
print(len(affirmations))

# create mapping of unique characters
vocab = sorted(set(affirmations))
char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)

# confirm functions working properly
print(char2index["a"])
print(index2char[char2index["a"]])


# converts string to int array
def text2int(text):
    arr = []
    for char in text:
        arr.append(char2index[char])
    return arr


# converts integer array to char array
def intArray2Text(intArr):
    arr = []
    for int in intArr:
        arr.append(index2char[int])
    return arr


# confirm functions work
print(text2int(affirmations[100:120]))
print(intArray2Text(text2int(affirmations[100:120])))

# generate datasets
idsDatasets = tf.data.Dataset.from_tensor_slices(text2int(affirmations))

# generate examples using affirmations of length 100
seqLength = 100
examples_per_epoch = len(affirmations) // (seqLength + 1)
sequences = idsDatasets.batch(seqLength + 1, drop_remainder=True)


# splits sequences of characters into a given and a prediction goal
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text


# map this split to all of the generated sequences
dataset = sequences.map(split_input_target)

# check datasets are split properly
for x, y in dataset.take(2):
    print(intArray2Text(x), intArray2Text(y))

BATCH_SIZE = 64
# Buffer size to shuffle the dataset
BUFFER_SIZE = 10000


# function that creates the neural network model
def build_model(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabSize, embeddingDim,
                                  batch_input_shape=[batchSize, None]),
        tf.keras.layers.LSTM(rnnUnits,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocabSize)
    ])
    return model


VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# create model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()


# loss function to determine difference between actual and expected output
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# compile model using our loss function
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                         save_weights_only=True)

# begin training model
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
history = model.fit(data, epochs=300, callbacks=[checkpoint_callback])

# reload model using 1 text input
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batchSize=1)

# load most recent model
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

affirmationLength = 100


# generates text based on given input
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 100

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension

        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return start_string + ''.join(text_generated)


inp = input("Type a starting string: ")
print(generate_text(model, inp))
