import tensorflow as tf
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import requests
import os
import psycopg2


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


# open affirmations
affirmations = open("Affirmations.txt", 'rb').read().decode(encoding='utf-8')

# create mapping of unique characters
vocab = sorted(set(affirmations))
char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)

DOC_SIZE = len(affirmations)
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# reload model using 1 text input
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batchSize=1)

# load most recent model
checkpoint_dir = './training_checkpoints'
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

affirmationLength = 100


# generates text based on given input
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 50

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2index[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 0.7

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

    return start_string + ''.join(text_generated).split("\n")[0]


# grabs random words from affirmations file
def choose_word():
    num = random.randrange(0, DOC_SIZE, 1)
    currChar = affirmations[num]

    while currChar != " " and num > -1:
        currChar = affirmations[num]
        num -= 1

    wordStart = num + 2
    num += 2
    currChar = affirmations[num]

    while currChar != " " and num < DOC_SIZE:
        currChar = affirmations[num]
        num += 1
    wordEnd = num - 1
    return affirmations[wordStart:wordEnd]


# print(choose_word())
# inp = input("Type a starting string: ")

def splitText(text):
    mid = int(len(text) / 2)
    currChar = text[mid]
    while currChar != " " and mid < len(text):
        currChar = text[mid]
        mid += 1

    return text[0:mid].replace("\r", "").replace("\n", " "), text[mid:].replace("\r", "").replace("\n", " ")


# Setting up db connection
conn_string = "postgresql://tenheller:q8NCFp26rJ1wf1qf@free-tier.gcp-us-central1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full&sslrootcert=C:\\Users\\jmmor\\AppData\\Roaming\\postgresql\\root.crt&options=--cluster%3Dtenler-test-5472"
connection = psycopg2.connect(os.path.expandvars(conn_string))
conn = connection.cursor()


# IMAGE GENERATION
def link(conn_string, animeid, backid, affid):
    conn.execute("insert into connections_test VALUES (DEFAULT, " + str(animeid) + ", " + str(affid) + ", " + str(
        backid) + ", DEFAULT)")
    connection.commit()


def getimg(conn, imgid, isanime):
    if isanime:
        tablename = "images_test"
    else:
        tablename = "backgrounds_test"
    conn.execute("select imgurl from " + tablename + " where id = " + str(imgid))
    imgurl = conn.fetchone()[0]
    return imgurl


def selectimg(conn, isanime):  # isanime is boolean, returns id of an image
    if isanime:
        tablename = "images_test"
        idtype = "img_id"
    else:
        tablename = "backgrounds_test"
        idtype = "background_id"
    conn.execute(
        "select imgtable.id from " + tablename + " as imgtable left join connections_test on imgtable.id = connections_test." + idtype + " group by imgtable.id order by count(distinct connections_test.id) asc")
    imgid = conn.fetchone()[0]
    return imgid


def makeaff(conn, text):  # text is string, inserts row into affirmations_test
    conn.execute("insert into affirmations_test VALUES (DEFAULT, \'" + text + "\', DEFAULT)")
    connection.commit()
    conn.execute("select id from affirmations_test order by created_at desc limit 1")
    return conn.fetchone()[0]


imagenames = ["primary.png", "secondary.png"]


def downloader(urls, names):
    imagenumber = 0
    for url in urls:
        response = requests.get(url, allow_redirects=True)
        file = open(names[imagenumber], "wb")
        file.write(response.content)
        file.close()
        imagenumber = imagenumber + 1


def transparency(image, name):
    rgba = image.convert("RGBA")
    data = rgba.getdata()
    newpicture = []
    for pixel in data:
        if (pixel[0] == pixel[1] == pixel[2]) and (pixel[0] > 180):
            newpicture.append((255, 255, 255, 0))
        else:
            newpicture.append(pixel)
    rgba.putdata(newpicture)
    rgba.save(name)


def drawTextWithHalo(img, position, text, font, col, haloCol):
    ImageDraw.Draw(img).text((position[0] - 5, position[1] - 5), text, font=font, fill=haloCol)
    ImageDraw.Draw(img).text((position[0] + 5, position[1] + 5), text, font=font, fill=haloCol)
    ImageDraw.Draw(img).text(position, text, font=font, fill=col, stroke_width=1, stroke_fill=(0, 0, 0))


def imageCombiner(urls, names, text):
    downloader(urls, names)
    primary = Image.open(names[0])
    primary.convert("RGBA")
    primary.putalpha(255)
    secondary = Image.open(names[1])
    widthpri, heightpri = primary.size
    widthsec, heightsec = secondary.size
    area = (int(widthpri / 2 - widthsec / 2), int(heightpri / 2 - heightsec / 2),
            int(widthpri / 2 + widthsec / 2), int(heightpri / 2 + heightsec / 2))
    secondary.resize(((int(widthpri / 2), int(heightpri / 2))))
    blank = Image.new("RGBA", primary.size, 255)

    # blank.show()
    blank.paste(secondary, area)
    secondary = blank
    transparency(secondary, "secondary.png")
    secondary = Image.open("secondary.png")

    # secondary.show()
    secondary.convert("RGBA")
    primary = Image.alpha_composite(primary, secondary)
    fontSize = int(widthpri / len(text[0]) * 2)

    color = (random.randrange(0, 255, 1), random.randrange(0, 255, 1), random.randrange(0, 255, 1), 255)
    colorBlur = (color[0], color[1], color[2], 128)
    font = ImageFont.truetype("impact.ttf", fontSize)
    textWidthTop, textHeightTop = ImageDraw.Draw(primary).textsize(text[0], font=font)
    textWidthBot, textHeightBot = ImageDraw.Draw(primary).textsize(text[1], font=font)

    drawTextWithHalo(primary, ((widthpri - textWidthTop) / 2, 10),
                     text[0], font, color, colorBlur)
    drawTextWithHalo(primary, ((widthpri - textWidthBot) / 2, heightpri - textHeightBot - 10),
                     text[1], font, color, colorBlur)

    primary.show()


while True:
    broken = False
    try:
        # Selects images
        animeimgid = selectimg(conn, True)
        backimgid = selectimg(conn, False)
        # Set equal to something -> returns anime img url
        urls = [getimg(conn, backimgid, False), getimg(conn, animeimgid, True)]
        randWord = choose_word()
        print(randWord, urls)

        afftext = generate_text(model, randWord)
        # Generate Affirmation and save in afftext

        # adds logging row to affirmations_test
        affid = makeaff(conn, afftext)

        # makes row in connections_test
        link(conn, animeimgid, backimgid, affid)

        imageCombiner(urls, imagenames, splitText(afftext))
        break

    except Exception as e:
        pass
