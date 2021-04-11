import numpy as np

class ErrorCalculating:
    def __init__(self):
        pass

    def _levenshtein_distance(self, ref, hyp):
        """Levenshtein distance is a string metric for measuring the difference
        between two sequences. Informally, the levenshtein disctance is defined as
        the minimum number of single-character edits (substitutions, insertions or
        deletions) required to change one word into the other. We can naturally
        extend the edits to word level when calculate levenshtein disctance for
        two sentences.
        """
        m = len(ref)
        n = len(hyp)

        # special case
        if ref == hyp:
            return 0
        if m == 0:
            return n
        if n == 0:
            return m

        if m < n:
            ref, hyp = hyp, ref
            m, n = n, m

        # use O(min(m, n)) space
        distance = np.zeros((2, n + 1), dtype=np.int32)

        # initialize distance matrix
        for j in range(0, n + 1):
            distance[0][j] = j

        # calculate levenshtein distance
        for i in range(1, m + 1):
            prev_row_idx = (i - 1) % 2
            cur_row_idx = i % 2
            distance[cur_row_idx][0] = i
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
                else:
                    s_num = distance[prev_row_idx][j - 1] + 1
                    i_num = distance[cur_row_idx][j - 1] + 1
                    d_num = distance[prev_row_idx][j] + 1
                    distance[cur_row_idx][j] = min(s_num, i_num, d_num)

        return distance[m % 2][n]

    def char_errors(self, reference, hypothesis, ignore_case=False, remove_space=False):
        """Compute the levenshtein distance between reference sequence and
        hypothesis sequence in char-level.
        :param reference: The reference sentence.
        :type reference: basestring
        :param hypothesis: The hypothesis sentence.
        :type hypothesis: basestring
        :param ignore_case: Whether case-sensitive or not.
        :type ignore_case: bool
        :param remove_space: Whether remove internal space characters
        :type remove_space: bool
        :return: Levenshtein distance and length of reference sentence.
        :rtype: list
        """
        if ignore_case == True:
            reference = reference.lower()
            hypothesis = hypothesis.lower()

        join_char = ' '
        if remove_space == True:
            join_char = ''

        reference = join_char.join(filter(None, reference.split(' ')))
        hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

        edit_distance = self._levenshtein_distance(reference, hypothesis)
        return float(edit_distance), len(reference)

    def cer(self, reference, hypothesis, ignore_case=False, remove_space=False):
        """Calculate charactor error rate (CER). CER compares reference text and
        hypothesis text in char-level. CER is defined as:
        .. math::
            CER = (Sc + Dc + Ic) / Nc
        where
        .. code-block:: text
            Sc is the number of characters substituted,
            Dc is the number of characters deleted,
            Ic is the number of characters inserted
            Nc is the number of characters in the reference
        We can use levenshtein distance to calculate CER. Chinese input should be
        encoded to unicode. Please draw an attention that the leading and tailing
        space characters will be truncated and multiple consecutive space
        characters in a sentence will be replaced by one space character.
        :param reference: The reference sentence.
        :type reference: basestring
        :param hypothesis: The hypothesis sentence.
        :type hypothesis: basestring
        :param ignore_case: Whether case-sensitive or not.
        :type ignore_case: bool
        :param remove_space: Whether remove internal space characters
        :type remove_space: bool
        :return: Character error rate.
        :rtype: float
        :raises ValueError: If the reference length is zero.
        """
        edit_distance, ref_len = self.char_errors(reference, hypothesis, ignore_case,
                                            remove_space)

        if ref_len == 0:
            raise ValueError("Length of reference should be greater than 0.")

        cer = float(edit_distance) / ref_len
        return cer
