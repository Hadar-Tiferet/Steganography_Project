from PIL import Image
import numpy as np
import secrets
from os.path import splitext, exists
from sys import argv


def hide(pixel_array, message, random_start=False):
    """
    hide a given message within the pixel RGB values of a given image file
    :param pixel_array: an array of values, corresponding to RGB values of pixels in an image file
    :param message: message to be hidden within the pixel array
    :param random_start: define the starting position of the hidden message, true - a random index; false - index 0
    :return: a modified pixel array, containing the message hidden within
    """
    # save array dimensions for reconstruction
    array_rows = len(pixel_array)
    array_columns = len(pixel_array[0])
    array_values = len(pixel_array[0][0])

    flat_array = pixel_array.flatten()
    # get ascii representation of the chars in message
    message_values = np.array(list(bytes(message, 'ascii')), dtype=np.uint8)
    # get binary representation of the message, in big endian
    message_binary = np.unpackbits(message_values, bitorder='big')
    index = 0  # set the starting point of insertion into the pixel array
    extra_space = len(flat_array) - len(message_binary)
    if extra_space < 0:
        raise Exception("Error: given message is too large for the given image file")

    if random_start:
        # if user has selected to start hiding the message at a random value - generate a valid random starting index
        index = secrets.randbelow(extra_space + 1)

    for value in message_binary:
        flat_array[index] = modify_bit(flat_array[index], value)
        index += 1

    return np.reshape(flat_array, (array_rows, array_columns, array_values))


def image_to_array(image_filename):
    """
    Read an image into an RBG pixel array, using the PIL package
    :param image_filename: name of the image file to read data from
    :return: an array of values corresponding to the RGB data of each pixel in the image
    """
    try:
        with Image.open(image_filename) as img:
            return np.asarray(img)
    except Exception as exception:
        print("encountered an error reading the image file: " + image_filename)
        raise exception


def array_to_image(pixel_array, image_name, image_extension):
    """
    save an image based on an array of RGB values per pixel, using the PIL package
    :param pixel_array: an array of values, corresponding to RGB values of pixels in an image file
    :param image_name: name of the original image file
    :param image_extension: extension of the name of the original image file
    """
    edited = Image.fromarray(pixel_array)
    edited_image_filename = image_name + "_hidden" + image_extension
    try:
        edited.save(edited_image_filename)
    except Exception as exception:
        print("encountered an error writing the edited image file: " + edited_image_filename)
        raise exception


def modify_bit(number, bit, position=0):
    """
    modify a number by specifying the value of a specific bit in its binary representation
    :param number: value to be modified
    :param bit: value of the bit to be enforced
    :param position: position, from the LSB, of the bit to be enforced
    :return: the modified number, after setting the value of the given bit
    """
    mask = 1 << position
    return (number & ~mask) | ((bit << position) & mask)


def extract_message(message):
    """
    return the message, whether it is given directly or through the name of the containing text file
    :param message: the potential message / text file name to be checked
    :return: the message itself
    """
    if exists(message) and splitext(message)[1] == ".txt":
        # message is a text file name, read the requested message from it
        try:
            with open(message) as file:
                message = file.read()
        except OSError as exception:
            print("encountered an error reading the message text file: " + message)
            raise exception
    return message


def main(image_file, message, random_start=False):
    try:
        message = extract_message(message)
        pixel_array = image_to_array(image_file)
        edited_array = hide(pixel_array, message, random_start)
        image_name, image_extension = splitext(image_file)
        array_to_image(edited_array, image_name, image_extension)
    except Exception as exception:
        print("Program has encountered an error and failed.")
        print(exception)


if __name__ == "__main__":
    default_image = "hidden.png"
    default_message = "this is a default hidden message"
    randomize = False
    if argv[-1] == '$r':
        randomize = True
        argv.pop(-1)
    argv.pop(0)

    if len(argv) == 0:
        main(default_image, default_message, randomize)
    elif len(argv) == 1:
        main(argv[0], default_message, randomize)
    else:
        input_message = " ".join(argv[1:])
        main(argv[0], input_message, randomize)
