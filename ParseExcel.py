import pandas

df = pandas.read_csv("first 100 affirmations - Sheet1.csv")
w = open("Affirmations.txt", "w")

affirmations = df["Affirmation:"]
for affirmation in affirmations:
    w.write(affirmation + "\n")

w.close()
