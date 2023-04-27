

class CodeTranslationTemplate:
    @staticmethod
    def get_instruction_1(x, src_lang, tgt_lang):
        instruction = f"Translate following {src_lang} code to {tgt_lang} code. " \
                      f"The {src_lang} code is: \n{x}" \
                      f"Only return the {tgt_lang} code."
        return instruction

    @staticmethod
    def get_instruction_2(x, src_lang, tgt_lang):
        instruction = f'You are given the task of translating {src_lang} code to {tgt_lang} code.' \
                      f'The input {src_lang} code is: \n{x}' \
                      f'Only return the {tgt_lang} code.'
        return instruction

    @staticmethod
    def get_instruction_3(x, src_lang, tgt_lang):
        instruction = f"Transfer following {src_lang} code to {tgt_lang} code. " \
                      f"The {src_lang} code is: \n {x}" \
                      f"Only return the {tgt_lang} code."
        return instruction


class CodeSummarizationTemplate:
    @staticmethod
    def get_instruction_1(x, src_lang, tgt_lang):
        instruction = f'Summarize the following {src_lang} code with natural language description. ' \
                      f'The {src_lang} code is: \n{x}'
        return instruction

    @staticmethod
    def get_instruction_2(x, src_lang, tgt_lang):
        instruction = f'You are given the task of generating comments for {src_lang} code. ' \
                      f'The {src_lang} code is: \n{x}'
        return instruction

    @staticmethod
    def get_instruction_3(x, src_lang, tgt_lang):
        instruction = f'Generate comments the following {src_lang} code. ' \
                      f'The {src_lang} code is: \n{x}'
        return instruction


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


ManuPrmoptClassList = [
    CodeTranslationTemplate,
    CodeSummarizationTemplate,
    NL2SQLTemplate,
]