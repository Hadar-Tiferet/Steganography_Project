from PIL import Image
import numpy as np
import concurrent.futures
import copy
import math
from sys import argv


def image_to_array(filename):
    """
    Read an image into an RBG pixel array, using the PIL package
    :param filename: name of the image file to read data from
    :return: an array of values corresponding to the RGB data of each pixel in the image
    """
    try:
        with Image.open(filename) as img:
            return np.asarray(img)
    except Exception as exception:
        print("encountered an error reading the image file: " + filename)
        raise exception


def decode(pixel_array, common_words_dict, lowest_bits=3, print_results=False):
    """
    decode the most likely message hidden inside the pixel values array of an image file
    :param pixel_array: an array of values, corresponding to RGB values of pixels in an image file
    :param common_words_dict: a dictionary of words to check the given array against
    :param lowest_bits: number of bits to consider when decoding sentences, starting at the LSB
    :param print_results: true/false, specify if the function should print the most likely results
    :return: most likely sentence that was decoded from the pixel array, as a string
    """
    flat_array = pixel_array.flatten()
    # separate checks for each of the 8 sub arrays (shift of 1 bit)
    # account for each starting value for encryption possible
    byte = 8
    futures = list()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for shift in range(byte):
            # create parameters list
            if shift == 0:
                future = executor.submit(check_subarray, flat_array, common_words_dict, lowest_bits)
            else:
                future = executor.submit(
                    check_subarray, flat_array[shift:-((-shift) % byte)], common_words_dict, lowest_bits)
            futures.append(future)

    best_results = [(0, ""), (0, ""), (0, ""), (0, ""), (0, "")]
    for future in concurrent.futures.as_completed(futures):
        for result in future.result():
            if result[0] > best_results[0][0]:
                best_results[0] = result
                best_results = sorted(best_results)

    if print_results:
        for result in best_results:
            print("result: " + result[1] + "\nwith score of: " + str(result[0]))
    # return the most likely decoded string
    return best_results[-1][1]


def check_subarray(array, common_words_dict, lowest_bits):
    """
    given an array of values, decode the most likely message hidden inside
    :param array: an array of values, to be transformed into three arrays of bits (at location 0, 1, 2)
    and decoded into the most likely message hidden within
    :param common_words_dict: a dictionary of words to check the given array against
    :param lowest_bits: number of bits to consider when decoding sentences, starting at the LSB
    :return: most likely messages decoded from the given array, alongside scores
    """
    arrays = list()
    dictionaries = list()
    spaces = dict()
    mappings_array = 0
    for i in range(lowest_bits):
        # run through bits 0 to lowest_bits - 1
        current_array, current_dictionary, current_spaces = decode_bit(array, common_words_dict, i)
        arrays.append(current_array)
        dictionaries.append(current_dictionary)
        spaces.update(current_spaces)
        mappings_array = np.bitwise_or(mappings_array, current_array)
    index = 0
    text_areas = list()
    while index < len(mappings_array):
        # locate areas where words are found on any of the bit arrays
        if mappings_array[index] > 0:
            # a word has started
            starting_index = index
            while (index < len(mappings_array)) and ((mappings_array[index] > 0) or (index in spaces)):
                index += 1
            ending_index = index - 1
            possible_sentence_size = ending_index - starting_index + 1
            text_areas.append((possible_sentence_size, starting_index, ending_index))
        index += 1

    best_results = [(0, ""), (0, ""), (0, ""), (0, ""), (0, "")]
    for area in text_areas:
        results = find_best_sequences(dictionaries, spaces, area)
        for result in results:
            if result[0] > best_results[0][0]:
                best_results[0] = result
                best_results = sorted(best_results)

    return best_results


def find_best_sequences(word_dictionaries, spaces_dictionary, text_area):
    """
    find the most likely sentences formed by words with spaces in between, in the location indicated by text_area
    :param word_dictionaries: dictionaries listing valid words found in the arrays alongside their location in the array
    :param spaces_dictionary: a dictionary containing space characters found in the arrays, at the index they appear
    :param text_area: a tuple of starting and ending index for the area to be checked for sentences in the array
    :return: most likely messages found in the array at the specified area, alongside scores
    """
    best_results = [(0, ""), (0, ""), (0, ""), (0, ""), (0, "")]
    sequences_lists = list()
    sequence_dict = dict()
    current_index = text_area[1]
    end_index = text_area[2]
    while current_index <= end_index:
        for dictionary in word_dictionaries:
            if current_index in dictionary:
                if current_index not in sequence_dict:
                    # create a new list for possible words
                    sequence_dict[current_index] = list()
                for word, score in dictionary[current_index].items():
                    sequence_dict[current_index].append((score, word))
                sequence_dict[current_index] = sorted(sequence_dict[current_index], reverse=True)
        current_index += 1
    sequences_lists.append(copy.deepcopy(sequence_dict))

    ascending = True
    while ascending:
        new_sequence_dict = dict()
        for index in sequence_dict:
            current_list = list()
            for score, sequence in sequence_dict[index]:
                next_index = index + len(sequence)
                while next_index in spaces_dictionary:
                    # add the space character that exists after the known sequence
                    sequence += spaces_dictionary[next_index]
                    next_index += 1
                    # add to current list any combinations of the known sequence, followed by one or more space chars
                    # and then another sequence of the same level
                    if next_index in sequence_dict:
                        for next_score, next_sequence in sequence_dict[next_index]:
                            current_list.append((score + next_score, sequence + next_sequence))
            if current_list:
                # the list of updated sequences is not empty - update findings
                new_sequence_dict[index] = copy.deepcopy(
                    sorted(current_list, reverse=True)[:math.ceil(len(current_list)/2)])

        sequence_dict = new_sequence_dict
        if sequence_dict:
            # current sequence dictionary is not empty - add it to the list
            sequences_lists.append(copy.deepcopy(sequence_dict))
        else:
            # current sequence dictionary is empty - finish ascending
            ascending = False

    if sequences_lists:
        sequence_dict = copy.deepcopy(sequences_lists[-1])
        sequences_lists = sequences_lists[:-1]

    while sequences_lists:
        # iterate as long as there are sequence dictionaries left in the list
        current_dict = sequences_lists[-1]
        for index in sequence_dict:
            current_list = list()
            for score, sequence in sequence_dict[index]:
                next_index = index + len(sequence)
                while next_index in spaces_dictionary:
                    # add the space character that exists after the known sequence
                    sequence += spaces_dictionary[next_index]
                    next_index += 1
                    # add to current list any combinations of the known sequence, followed by one or more space chars
                    # and then another sequence of the same level
                    if next_index in current_dict:
                        for next_score, next_sequence in current_dict[next_index]:
                            current_list.append((score + next_score, sequence + next_sequence))
            if current_list:
                # the list of updated sequences is not empty - update findings
                sequence_dict[index] = copy.deepcopy(
                    sorted(current_list, reverse=True)[:math.ceil(len(current_list)/2)])
        sequences_lists = sequences_lists[:-1]
    for index in sequence_dict:
        for score, sequence in sequence_dict[index]:
            if score > best_results[0][0]:
                # insert the entry into the best scores list
                best_results[0] = (score, sequence)
                best_results = sorted(best_results)

    return best_results


def decode_bit(array, common_words_dict, bit, order='big'):
    """
    given an array of values and a bit to focus on, index words and spaces formed from the specific bit in the array,
    alongside an array mapping the values of the original array
    :param array: an array of values to be digested
    :param common_words_dict: a dictionary of words to check the given array against
    :param bit: location of the bit, from the LSB to focus the search for words on
    :param order: a choice to interpret values as big endian of little endian
    :return: a tuple of an array mapping text areas and spaces, a dictionary of words found at various indexes and
    a dictionary of spaces found
    """
    binary_array = separate_bit(array, bit)
    values_array = np.packbits(binary_array, bitorder=order)
    char_array = values_array.view('c')
    findings_dict = dict()
    space_dict = dict()
    # attempt to make the sentence building less taxing
    mapping_array = np.zeros(len(char_array), dtype='uint8')
    bit_value = pow(2, bit)
    index = 0
    while index < len(char_array):
        char = char_array[index]
        index_dict = dict()
        if char.isalpha():
            # current index holds a character
            current_string = char.decode()
            # check if the given letter is a common word on its own
            if current_string.lower() in common_words_dict:
                index_dict[current_string] = common_words_dict[current_string.lower()]
                mapping_array[index] = bit_value
            co_index = index + 1
            while (co_index < len(char_array)) and (char_array[co_index].isalpha() or char_array[co_index] == b"'"):
                current_string += char_array[co_index].decode()
                if current_string.replace("'", "").lower() in common_words_dict:
                    index_dict[current_string] = common_words_dict[current_string.replace("'", "").lower()]
                    mapping_array[index:co_index] = bit_value
                co_index += 1  # array from index to co_index might contain a valid word
            # at the end of a word, there might be valid markings
            unmarked_string = current_string.lower()
            unmarked_index = co_index - 1
            if unmarked_string in common_words_dict:
                while (co_index < len(char_array)) and char_array[co_index] in (b'.', b',', b"?", b'!', b':'):
                    current_string += char_array[co_index].decode()
                    # add markings at the end of a word as additional valid words
                    index_dict[current_string] = common_words_dict[unmarked_string] + co_index - unmarked_index
                    mapping_array[co_index] = bit_value
                    co_index += 1
        elif char in (b' ', b'\t', b'\n'):
            # current index holds a space character
            space_dict[index] = char.decode()
        if index_dict:
            # word dictionary for current index is not empty
            # consider sorting the dict in reverse to have the largest possible values first
            findings_dict[index] = dict(sorted(index_dict.items(), reverse=True))

        index += 1
    return mapping_array, findings_dict, space_dict


def separate_bit(array, bit):
    """
    return an array of values indicating whether the specified bit is set in the original array entries
    :param array: an array of values to be tested
    :param bit: location of the bit to be tested, starting from the Least Significant Bit
    :return: an array of the specified bit value for the entries of the original array
    """
    mask = pow(2, bit)
    return (array & mask) >> bit


def normalize_score(score, min_value, max_value, desired_min, desired_max):
    """
    normalize every given score to fit between desired input values
    :param score: score to be normalized
    :param min_value: minimum score to be considered
    :param max_value: maximum score to be considered
    :param desired_min: normalized minimum desired for a valid score
    :param desired_max: normalized maximum desired for a valid score
    :return: a score, normalized between the two desired extremes
    """
    return (desired_min + (score - min_value)*(desired_max - desired_min)) / (max_value - min_value)


def calculate_word_score(word, value):
    return value + len(word)


def build_word_dictionary(dictionary_file, desired_min, desired_max):
    """
    build a word dictionary from a file, containing pairs of words and scores
    :param dictionary_file: name of the file to extract the dictionary data from
    :param desired_min: desired minimum value for a normalized dictionary entry
    :param desired_max: desired maximum value for a normalized dictionary entry
    :return: a dictionary of words alongside normalized scores
    """
    common_words_dict = dict()
    try:
        with open(dictionary_file, 'r') as file:
            lines = file.readlines()
            max_score = min_score = int(lines[0].split()[1])
            for line in lines:
                word, value = line.split()
                word = word.strip().lower()
                value = int(value)
                common_words_dict[word] = value
                if value > max_score:
                    max_score = value
                if value < min_score:
                    min_score = value
    except OSError as exception:
        print("encountered an error reading the dictionary file: " + dictionary_file)
        raise exception
    except ValueError as exception:
        print("encountered an error while building up common words dictionary from: " + dictionary_file +
              "\nDictionary file should contain lines of a single word alongside a score for each word")
        raise exception
    if max_score == min_score:
        raise Exception("encountered an error while building up a common words dictionary from: " + dictionary_file +
              "\nA dictionary file should have a meaningful score for each word, not the same score for every word")
    # normalize word values in the dictionary
    for word in common_words_dict:
        normalized_value = normalize_score(common_words_dict[word], min_score, max_score, desired_min, desired_max)
        common_words_dict[word] = calculate_word_score(word, normalized_value)

    return common_words_dict


def write_result(filename, string):
    try:
        with open(filename, 'w') as file:
            file.write(string)
    except OSError as exception:
        print("encountered an error writing to the result file: " + filename)
        raise exception


def main(image_file="hidden.png", export_file="205492507.txt", dictionary_file="word_dictionary.txt",
         print_results=False):
    try:
        common_words_dictionary = build_word_dictionary(dictionary_file, 0, 2)
        image_array = image_to_array(image_file)
        decoded_string = decode(image_array, common_words_dictionary, print_results=print_results)
        write_result(export_file, decoded_string)
    except Exception as exception:
        print("Program has encountered an error and failed.")
        print(exception)


if __name__ == "__main__":
    print_results_to_screen = False
    if argv[-1] == '$p':
        print_results_to_screen = True
        argv.pop(-1)
    argv.pop(0)

    if len(argv) == 0:
        # run the program with default values
        main("hidden.png", print_results=print_results_to_screen)
    elif len(argv) == 1:
        # run the program with only image file given as input
        main(image_file=argv[0], print_results=print_results_to_screen)
    elif len(argv) == 2:
        # run the program with image file and export file given as input
        main(argv[0], argv[1], print_results=print_results_to_screen)
    else:
        # run the program with all arguments given from input
        main(argv[0], argv[1], argv[2], print_results=print_results_to_screen)
