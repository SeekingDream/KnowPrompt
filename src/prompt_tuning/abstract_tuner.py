

class AbstractTuner:
    def __init__(self, train_data, eval_data, init_prompt, llm_func, config):
        self.train_data = train_data
        self.eval_data = eval_data

        self.init_prompt = init_prompt
        self.best_prompt = init_prompt
        self.llm_func = llm_func
        self.config = config

    def score_func(self, prompt):
        # assign a score to this prompt
        raise NotImplementedError

    def tune(self):
        pass

    def evaluate_prompt(self):
        pass

