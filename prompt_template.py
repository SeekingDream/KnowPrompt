

class CodeTranslationTemplate:
    def get_instruction_1(self, x, src_lang, tgt_lang):
        instruction = f'Translate the following {src_lang} code to {tgt_lang} code. The {src_lang} code is: ' +  x                  # TODO
        return instruction + x

    def get_instruction_2(self, x, src_lang, tgt_lang):
        instruction = ''
        return instruction + x

    def get_instruction_3(self, x, src_lang, tgt_lang):
        instruction = ''
        return instruction + x


class CodeSummarizationTemplate:
    def get_instruction_1(self, x):
        instruction = ''
        return instruction + x

    def get_instruction_2(self, x):
        instruction = ''
        return instruction + x

    def get_instruction_3(self, x):
        instruction = ''
        return instruction + x


class NL2SQLTemplate:
    def get_instruction_1(self, x):
        instruction = ''
        return instruction + x

    def get_instruction_2(self, x):
        instruction = ''
        return instruction + x

    def get_instruction_3(self, x):
        instruction = ''
        return instruction + x
