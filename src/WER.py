import numpy as np

def calculate_wer(reference, hypothesis):
    """
    Calculate the Word Error Rate (WER).
    
    :param reference: The correct transcript as a string.
    :param hypothesis: The transcribed text as a string.
    :return: WER as a float.
    """
    reference = reference.split()
    hypothesis = hypothesis.split()
    d = np.zeros((len(reference) + 1) * (len(hypothesis) + 1), dtype=np.uint8)
    d = d.reshape((len(reference) + 1, len(hypothesis) + 1))
    for i in range(len(reference) + 1):
        for j in range(len(hypothesis) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    for i in range(1, len(reference) + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    return d[len(reference)][len(hypothesis)] / float(len(reference))
