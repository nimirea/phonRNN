import torch


# feature trees

class OrigFeatures(object):
    # "kitchen sink" approach
    def __init__(self, parser, set_unset=False):
        # complete list of features:
        self.feature_list = ['syll', 'cons', 'appx', 'son', 'nasal', 'cont', 'voice', 'strid', 'lat', 'sg', 'cg', 'dr', 'LAB', 'COR', 'GLOT', 'DOR', 'round', 'ant', 'dist', 'high', 'low', 'back', 'diph', 'wide', 'atr', 'EOW', 'start']

        self.parser = parser

        self.set_unset = set_unset

        self.feature_idx = {}
        for f, feature in enumerate(self.feature_list):
            self.feature_idx[feature] = f


    def to_feature_vec(self, symbol):

        parser = self.parser
        result = [0] * len(self.feature_list)

        # TODO problem: how to encode these? as it stands, they are "closer" to some phones but not others; in point of fact they should probably be perfectly in the middle of the vector space
        result[self.feature_idx['EOW']] = -1
        result[self.feature_idx['start']] = -1
        if symbol == '\n':
            result[self.feature_idx['EOW']] = 1
        elif symbol == '<s>':
            result[self.feature_idx['start']] = 1
        else:
            # strip out stress mark, to be used later
            if (len(symbol) == 3):
                phone = symbol[:2]
            else:
                phone = symbol

            # major class features
            result[self.feature_idx['syll']] = 1 if (phone in parser.vowels | set(["ER"])) else -1
            result[self.feature_idx['cons']] = -1 if phone in (parser.glides | parser.vowels) else 1
            result[self.feature_idx['appx']] = 1 if result[self.feature_idx['cons']] == -1 or phone in parser.liquids else -1
            result[self.feature_idx['son']] = 1 if result[self.feature_idx['appx']] == 1 or phone in parser.nasals else -1

            # laryngeal features
            result[self.feature_idx['voice']] = -1 if (phone in parser.voiceless) else 1
            result[self.feature_idx['sg']] = 1 if phone in parser.glottal else -1
            result[self.feature_idx['cg']] = -1 # all sounds [-cg] for English

            # manner features
            result[self.feature_idx['nasal']] = 1 if phone in parser.nasals else -1
            result[self.feature_idx['cont']] = 1 if phone in (parser.fricatives | parser.retroflex | parser.glides | parser.vowels) else -1
            result[self.feature_idx['lat']] = 1 if phone in parser.lateral else -1
            result[self.feature_idx['strid']] = 1 if phone in ((parser.affricates | parser.fricatives) - parser.dental) else -1
            if phone in parser.affricates:
                result[self.feature_idx['dr']] = 1
            elif phone in (parser.stops | parser.nasals):
                result[self.feature_idx['dr']] = -1

            # place features:
            # LAB
            if (phone in parser.bilabial | parser.labiodental):
                result[self.feature_idx['LAB']] = 1
                result[self.feature_idx['round']] = 1 if phone in parser.rounded else -1
            else:
                result[self.feature_idx['LAB']] = -1

            # COR
            if (phone in parser.alveolar | parser.postalveolar | parser.dental):
                result[self.feature_idx['COR']] = 1
                result[self.feature_idx['ant']] = 1 if phone in (parser.alveolar | parser.dental) else -1
                result[self.feature_idx['dist']] = -1 if phone in (parser.alveolar | parser.retroflex) else 1
            else:
                result[self.feature_idx['COR']] = -1

            # DOR
            if (phone in parser.vowels | parser.palatal | parser.velar):
                result[self.feature_idx['DOR']] = 1
                result[self.feature_idx['back']] = 1 if phone in parser.back_vowels | parser.velar | parser.backcent_vowels else -1
                result[self.feature_idx['high']] = 1 if phone in (parser.velar | parser.palatal | parser.affricates | parser.high_vowels) else -1
                result[self.feature_idx['low']] = 1 if phone in parser.low_vowels else -1

                # vowel-only features
                if phone in parser.vowels:
                    # movement features
                    result[self.feature_idx['diph']] = 1 if phone in parser.diphthongs else -1
                    result[self.feature_idx['wide']] = 1 if phone in parser.wide_movement else -1

                    result[self.feature_idx['round']] = 1 if phone in parser.rounded else -1
                    result[self.feature_idx['atr']] = 1 if phone in parser.tense_vowels else -1

            else:
                result[self.feature_idx['DOR']] = -1

            result[self.feature_idx['GLOT']] = 1 if (phone in parser.glottal) else -1

        # add in the set (1) unset (-1) nodes:
        if self.set_unset:
            set_unset_vec = [abs(x) * 2 - 1 for x in result]
            result += set_unset_vec

        return torch.FloatTensor(result)

class FutrellFeatures(object):
    # based on Futrell 2017
    def __init__(self, parser, set_unset = False):
        # complete list of features:
        self.feature_list = ['duration', 'laryngeal', 'nasal', 'suprasegmental', 'manner_V', 'manner_nasal', 'manner_stop', 'manner_fric', 'manner_appx', 'manner_latappx', 'backness_Z', 'backness_Y', 'backness_X', 'backness_W', 'backness_U', 'height_E', 'height_D', 'height_Q', 'height_B', 'height_A', 'rounding', 'cplace_B', 'cplace_F', 'cplace_D', 'cplace_T', 'cplace_S', 'cplace_R', 'cplace_J', 'cplace_K', 'cplace_H', 'art2', 'lat', 'EOW', 'start']

        self.parser = parser

        self.set_unset = set_unset

        self.feature_idx = {}
        for f, feature in enumerate(self.feature_list):
            self.feature_idx[feature] = f

    def to_feature_vec(self, symbol):

        parser = self.parser

        result = [0] * len(self.feature_list)

        # TODO problem: how to encode these? as it stands, they are "closer" to some phones but not others; in point of fact they should probably be perfectly in the middle of the vector space
        result[self.feature_idx['EOW']] = -1
        result[self.feature_idx['start']] = -1
        if symbol == '\n':
            result[self.feature_idx['EOW']] = 1
        elif symbol == '<s>':
            result[self.feature_idx['start']] = 1
        else:
            # strip out stress marks for defining most features
            phone = parser.no_stress(symbol)

            # features common to all phones:
            result[self.feature_idx['duration']] = 1 if parser.is_long(phone) else -1
            result[self.feature_idx['laryngeal']] = -1 if (phone in parser.voiceless) else 1
            result[self.feature_idx['nasal']] = 1 if (phone in parser.nasals) else -1

            # manner nodes
            # set all manner nodes to -1
            for i in self.feature_list:
                if "manner_" in i:
                    result[self.feature_idx[i]] = -1
            # vowel tree
            if phone in parser.vowels:
                result[self.feature_idx['manner_V']] = 1

                # zero out nodes with multiple values
                for i in self.feature_list:
                    if "height_" in i or "backness_" in i:
                        result[self.feature_idx[i]] = -1

                # set height nodes
                if phone in parser.high_vowels:
                    result[self.feature_idx['height_E']] = 1
                elif phone in parser.openmid_vowels:
                    result[self.feature_idx['height_D']] = 1
                elif phone in parser.mid_vowels:
                    result[self.feature_idx['height_Q']] = 1
                elif phone in parser.closemid_vowels:
                    result[self.feature_idx['height_B']] = 1
                elif phone in parser.low_vowels:
                    result[self.feature_idx['height_A']] = 1

                # set backness nodes
                if phone in parser.front_vowels:
                    result[self.feature_idx['backness_Z']] = 1
                elif phone in parser.frontcent_vowels:
                    result[self.feature_idx['backness_Y']] = 1
                elif phone in parser.cent_vowels:
                    result[self.feature_idx['backness_X']] = 1
                elif phone in parser.backcent_vowels:
                    result[self.feature_idx['backness_W']] = 1
                elif phone in parser.back_vowels:
                    result[self.feature_idx['backness_U']] = 1

                # set rounding node
                if phone in parser.rounded:
                    result[self.feature_idx['rounding']] = 1
                else:
                    result[self.feature_idx['rounding']] = -1

                # set suprasegmental (stress) node
                result[self.feature_idx['suprasegmental']] = 1 if parser.is_stressed(symbol) else -1

            else:
                # set consonant manner
                if phone in parser.stops:
                    result[self.feature_idx['manner_stop']] = 1
                elif phone in parser.nasals:
                    result[self.feature_idx['manner_nasal']] = 1
                elif phone in parser.fricatives:
                    result[self.feature_idx['manner_fric']] = 1
                elif phone in (parser.glides | parser.retroflex):
                    result[self.feature_idx['manner_appx']] = 1
                elif phone in parser.lateral:
                    result[self.feature_idx['manner_latappx']] = 1

                # set nodes common to all consonants
                # zero out nodes with multiple values
                for i in self.feature_list:
                    if "cplace_" in i:
                        result[self.feature_idx[i]] = -1
                # consonant place
                if phone in parser.bilabial:
                    result[self.feature_idx['cplace_B']] = 1
                elif phone in parser.labiodental:
                    result[self.feature_idx['cplace_F']] = 1
                elif phone in parser.dental:
                    result[self.feature_idx['cplace_D']] = 1
                elif phone in parser.alveolar:
                    result[self.feature_idx['cplace_T']] = 1
                elif phone in parser.postalveolar - parser.retroflex:
                    result[self.feature_idx['cplace_S']] = 1
                elif phone in parser.retroflex:
                    result[self.feature_idx['cplace_R']] = 1
                elif phone in parser.palatal:
                    result[self.feature_idx['cplace_J']] = 1
                elif phone in parser.velar:
                    result[self.feature_idx['cplace_K']] = 1
                elif phone in parser.glottal:
                    result[self.feature_idx['cplace_H']] = 1

                # second articulator (at the moment, the velum is the only second articulator, so this only represents the velum)
                result[self.feature_idx['art2']] = 1 if (phone in parser.nasals) else -1

                # whether the consonant is a lateral or not:
                result[self.feature_idx['art2']] = 1 if (phone in parser.lateral) else -1

        # add in the set (1) unset (-1) nodes:
        if self.set_unset:
            set_unset_vec = [abs(x) * 2 - 1 for x in result]
            result += set_unset_vec

        return torch.FloatTensor(result)


# parsers

class ARPArser(object):
    def __init__(self):

        # SETS OF PHONES
        # NOTE: must add new phones if dealing wiith non-English language

        # manner classes
        self.vowels = set(["AA", "AE", "AH", "AO", "AW", "AY", "EH", "EY", "IH", "IY", "OW", "OY", "UH", "UW"])
        self.stops = set(["B", "D", "G", "K", "P", "T"])
        self.fricatives = set(["DH", "F", "HH", "S", "SH", "TH", "V", "Z", "ZH"])
        self.affricates = set(["CH", "JH"])
        self.nasals = set(["M","N","NG"])
        self.liquids = set(["L","R","ER"])
        self.glides = set(["Y","W"])
        self.retroflex = self.liquids - set(["L"])

        # place classes
        # labial place features
        self.bilabial = set(["B", "P", "M", "W"])
        self.labiodental = set(["F", "V"])
        self.rounded = set(["SH", "AA", "AO", "AW", "OW", "W", "OY", "UW", "UH"])
        # coronal place features
        self.dental = set(["TH", "DH"])
        self.alveolar = set(["N", "T", "D", "S", "Z", "L"])
        self.postalveolar = set(["CH", "JH", "SH", "ZH", "R", "ER"])
        self.palatal = set(["Y"])
        self.lateral = set(["L"])
        self.velar = set(["NG", "K", "G", "W"])
        self.glottal = set(["HH"])

        # vowel classes
        self.high_vowels = set(["IY", "IH", "UH", "UW"])
        self.low_vowels = set(["AE", "AA"])
        self.tense_vowels = set(["IY", "UW", "AE", "AH"]) # NOTE: since ARPAbet treats schwa and wedge the same, we're going to lump them in together and call them both central, not back
        self.diphthongs = set(["EY", "AY", "OW", "AW", "OY"])
        self.wide_movement = set(["AY", "AW", "OY"])
        self.closemid_vowels = set() # none to speak of
        self.mid_vowels = set(["AH"])
        self.openmid_vowels = set(["AO", "EH", "ER"])

        # place features
        # diphthongs classified by terminus, not origin
        self.front_vowels = set(["IY"])
        self.frontcent_vowels = set(["IH", "EH", "AE", "EY", "AY", "OY"])
        self.cent_vowels = set(["AH"])
        self.backcent_vowels = set(["UH", "AO", "AA", "AW", "OW"])
        self.back_vowels = set(["UW"])


        # voicing classes
        self.voiceless = set(["K", "P", "T", "F", "HH", "S", "SH", "TH"]) # everything else is voiced
        self.sibilants = set(["S", "Z", "SH", "ZH"])

    # return the unstressed version, for easier comparisons
    def no_stress(self, symbol):
        result = ''.join([i for i in symbol if not i.isdigit()])
        return (result)

    def is_stressed(self, symbol):
        return ("1" in symbol)

    def is_long(self, symbol):
        # no long vowels in this representation
        return False

class IPArser(object):
    def __init__(self):

        # SETS OF PHONES
        # NOTE: must add new phones if dealing wiith non-English language


        # manner classes
        self.vowels = set(['ɔ', 'ə', 'ʊ', 'æ', 'æɪ', 'oː', 'ɪ', 'æʊ', 'ɑː', 'ɛɪ', 'ɛ', 'uː', 'ʌ', 'iː', 'ɜː', 'əʊ', 'oɪ', 'oːɪ', 'ɑːɪ', 'o', 'e', 'ɔː', 'æː', 'ɑ', 'ɒː', 'a'])
        self.stops = set(["b", "d", "g", "k", "p", "t"])
        self.fricatives = set(['ð', "f", "h", "s", "ʃ", "θ", "v", "z", "ʒ", "x"])
        self.affricates = set(["t͡ʃ", "d͡ʒ"])
        self.nasals = set(["m","n","ŋ"])
        self.liquids = set(["l",'ɹ'])
        self.glides = set(["j","w"])
        self.retroflex = set(['ɹ'])

        # place classes
        # labial place features
        self.bilabial = set(["b", "p", "m", "w"])
        self.labiodental = set(["f", "v"])
        self.rounded = set(["ʃ", 'ɔ', 'æʊ', 'əʊ', 'oɪ', 'oːɪ', 'uː','ʊ', 'ɔ', 'ɔː', 'o', 'oː', 'ɒː'])
        # coronal place features
        self.dental = set(["θ", "ð"])
        self.alveolar = set(["n", "t", "d", "z", "s", "l"])
        self.postalveolar = set(["t͡ʃ", "d͡ʒ", "ʃ", "ʒ", 'ɹ'])
        self.palatal = set(["j"])
        self.lateral = set(["l"])
        self.velar = set(["ŋ", "k", "g", "w", "x"])
        self.glottal = set(["h"])

        # vowel classes
        self.high_vowels = set(['ʊ', 'ɪ', 'uː', 'iː'])
        self.low_vowels = set(['æ', 'ɑː', 'a', 'ɑ', 'ɒː', 'æː'])
        self.tense_vowels = set(['iː', 'uː', 'æ', 'ʌ'])
        self.diphthongs = set(['æɪ', 'æʊ', 'ɛɪ', 'əʊ', 'oɪ', 'oːɪ', 'ɑːɪ'])
        self.wide_movement = set(['æɪ', 'æʊ', 'oɪ', 'oːɪ'])
        # Futrell et al. vowel classes
        self.closemid_vowels = set(['oː', 'e', 'o'])
        self.mid_vowels = set(['ə'])
        self.openmid_vowels = set(['ɔ', 'ɛ', 'ʌ', 'ɜː'])


        # place features
        # dipththongs classified by terminus, not origin
        self.front_vowels = set(['iː', 'e'])
        self.frontcent_vowels = set(['ɪ', 'ɛ', 'æ', 'æɪ', 'ɛɪ', 'oɪ', 'oːɪ', 'ɑːɪ', 'a', 'æː'])
        self.cent_vowels = set(['ə', 'ɜː'])
        self.backcent_vowels = set(['ʊ', 'ʌ', 'ɔ', 'ɑː', 'æʊ', 'əʊ', 'ɔː', 'ɑ', 'ɒː'])
        self.back_vowels = set(['uː', 'oː', 'o'])


        # voicing classes
        self.voiceless = set(["k", "p", "t", "f", "h", "s", "ʃ", "θ", "t͡ʃ", "x"]) # everything else is voiced
        self.sibilants = set(["s", "ʃ", "z", "ʒ"])

    # return the unstressed version, for easier comparisons
    def no_stress(self, symbol):
        result = ''.join([i for i in symbol if not i.isdigit()])
        return (result)

    def is_stressed(self, symbol):
        return ("1" in symbol)

    def is_long(self, symbol):
        return ("ː" in symbol)
