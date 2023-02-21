def print_episode(df, season_episode, scene=None):
    if scene is None:
        df.apply(lambda x: print(f'[{x["speaker"]} - {x["scene"]}]: {x["line_text"]}') if x["season_episode"]==season_episode and not x["deleted"] else None, axis=1)
    else:
        df.apply(lambda x: print(f'[{x["speaker"]}]: {x["line_text"]}') if (x["season_episode"]==season_episode) & (x["scene"]==scene) else None, axis=1)

def get_word_appearance(df, word):
    return df[df.apply(lambda x: True if (word.lower() in x["line_text"].lower()) else False, axis=1)]
