class ConfirmTextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        confirm_model_char_map_str = """
        e 1
        n 2
        o 3
        s 4
        y 5
        <SPACE> 6
        """
        self.char_map = {}
        self.index_map = {}
        for line in confirm_model_char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(self.char_map[c])
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')

    def list_to_string(self, list):
        return ''.join(list)

class FoodNumberTextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
        food_number_char_map_str = """
        e 1
        f 2
        g 3
        h 4
        i 5
        n 6
        o 7
        r 8
        s 9
        t 10
        u 11
        v 12
        w 13
        x 14
        z 15
        <SPACE> 16
        """
        self.char_map = {}
        self.index_map = {}
        for line in confirm_model_char_map_str.strip().split('\n'):
            ch, index = line.split()
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch

    def text_to_int(self, text):
        """ Use a character map and convert text to an integer sequence """
        int_sequence = []
        for c in text:
            if c == ' ':
                ch = self.char_map['<SPACE>']
            else:
                ch = self.char_map[c]
            int_sequence.append(self.char_map[c])
        return int_sequence

    def int_to_text(self, labels):
        """ Use a character map and convert integer labels to an text sequence """
        string = []
        for i in labels:
            string.append(self.index_map[i])
        return ''.join(string).replace('<SPACE>', ' ')


    def list_to_string(self, list):
        return ''.join(list)