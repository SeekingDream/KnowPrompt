from src.prompt_tuning import GribsTuner

init_promp = 'Label the sentiment of the below tweet as positive or negative. The answer should be exact one word of \'positive\' or \'negative\'. The tweet is'
tuner = GribsTuner(None, None, init_promp, None, None)

tuner.tune()