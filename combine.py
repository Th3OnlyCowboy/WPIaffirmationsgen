from PIL import Image
import requests
urls = ["https://wallpaperaccess.com/full/622267.jpg",
        "https://www.pngfind.com/pngs/m/348-3486798_anime-boy-funny-manga-otaku-love-cute-cute.png"]
imagenames = ["primary.png", "secondary.png"]


def downloader(urls, names):
    imagenumber = 0
    for url in urls:
        response = requests.get(url, allow_redirects=True)
        file = open(names[imagenumber], "wb")
        file.write(response.content)
        file.close
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


def imagecombiner(urls, names):
    downloader(urls,names)
    primary = Image.open(names[0])
    primary.convert("RGBA")
    primary.putalpha(255)
    secondary = Image.open(names[1])
    widthpri, heightpri = primary.size
    widthsec, heightsec = secondary.size
    area = (int(widthpri/2 - widthsec/2), int(heightpri/2 - heightsec/2),
            int(widthpri/2 + widthsec/2), int(heightpri/2 + heightsec/2))
    secondary.resize((int(widthpri/2), int(heightpri/2)))
    blank = Image.new("RGBA", primary.size, 255)

    #blank.show()
    blank.paste(secondary, area)
    secondary = blank
    transparency(secondary, "secondary.png")
    secondary = Image.open("secondary.png")

    #secondary.show()
    secondary.convert("RGBA")
    primary = Image.alpha_composite(primary, secondary)
    primary.show()

imagecombiner(urls, imagenames)
