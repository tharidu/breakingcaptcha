import numpy as np

CHAR_VOCAB_SIZE = 36 # Each char in the word can either be a digit 0-9 or a letter a-z giving a total of 36 pssible characters.
WORD_SIZE =5


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
    # print word
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


def prediction_to_word(prediction_vector):
    """
    function to convert a prediction vector to captcha word
    :param prediction_vector: a [WORD_SIZE,CHAR_VOCAB_SIZE] np array of predictions
    :return: the string representing the word
    """
    b = np.zeros_like(prediction_vector)
    b[np.arange(len(prediction_vector)), prediction_vector.argmax(1)] = 1
    word_vector = np.reshape(b,WORD_SIZE*CHAR_VOCAB_SIZE)
    word = vec_to_word(word_vector)
    return word

def compare_predictions(predictions,labels):
    assert len(predictions == len(labels))
    print "True   | Predicted"
    for i,prediction in enumerate(predictions):
        label = labels[i]
        predicted_word = prediction_to_word(prediction)
        true_word = vec_to_word(label)
        print "{:7s}|{:10s}".format(true_word,predicted_word)


if __name__ == '__main__':
    np.set_printoptions(threshold='nan')
    print words_to_vec('12345')
    print vec_to_word( words_to_vec('12345'))