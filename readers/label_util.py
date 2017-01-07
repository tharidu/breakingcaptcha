import numpy as np

CHAR_VOCAB_SIZE = 36  # Each char in the word can either be a digit 0-9 or a letter a-z giving a total of 36 pssible characters.



def char_to_vec_pos(char):
    """
    Vector representation of length 36.
    First 10 positions are for digits 0-9
    Position 11-36 are for alphabets a-z
    :param char
    :return: returns the position of the char in the vector coding
    """
    ascii_val = ord(char)
    if ascii_val >= 48 and ascii_val<=57:
        return ascii_val-48
    if ascii_val >= 97 and ascii_val <=122:
        return (ascii_val-97)+10
    raise ValueError('Wrong character {}'.format(char))


def words_to_vec(word):
    """
    :param word: string of length 5 to be converted into vector
    :return: len*36 vector representation of word.
    """
    print word
    word_len = len(word)
    vec = np.zeros(word_len * CHAR_VOCAB_SIZE)
    #print len(vec)

    for i,char in enumerate(word):
        idx = (i*CHAR_VOCAB_SIZE)+char_to_vec_pos(char)
        vec[idx]=1
    return vec


def vec_to_word(vector):
    """
    :param vector: vector representation of word
    :return: string representation of the word
    """
    char_indices = vector.nonzero()[0]
    word = list()

    for idx in char_indices:
        vocab_idx = idx% CHAR_VOCAB_SIZE

        if vocab_idx < 10: # 0-9
            char_code = vocab_idx+ord('0')
        elif vocab_idx <= 35: # a-z
            char_code =  (vocab_idx - 10) + ord('a')
        else:
            raise ValueError("Incorrect character code")

        word.append(chr(char_code))

    return "".join(word)


if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    print words_to_vec('abbc7')
    print vec_to_word( words_to_vec('abbc7'))