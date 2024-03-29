from __future__ import print_function

import string
import numpy as np
import pandas as pd

translator = str.maketrans('', '', string.punctuation)
df = pd.read_csv("./data/lyrics.csv", sep="\t")
df.head()
pdf = pd.read_csv('./data/PoetryFoundationData.csv',quotechar='"')
pdf.head()


def split_text(x):
    text = x['lyrics']
    try:
        sections = text.split('\\n\\n')
    except AttributeError:
        return
    keys = {'Verse 1': np.nan, 'Verse 2': np.nan, 'Verse 3': np.nan, 'Verse 4': np.nan, 'Chorus': np.nan}
    single_text = []
    res = {}
    for s in sections:
        key = s[s.find('[') + 1:s.find(']')].strip()
        if ':' in key:
            key = key[:key.find(':')]

        if key in keys:
            single_text += [x.lower().replace('(', '').replace(')', '').translate(translator) for x in
                            s[s.find(']') + 1:].split('\\n') if len(x) > 1]

        res['single_text'] = ' \n '.join(single_text)
    return pd.Series(res)


df = df.join(df.apply(split_text, axis=1))
df.head()

pdf['single_text'] = pdf['Poem'].apply(lambda x: ' \n '.join([l.lower().strip().translate(translator) for l in x.splitlines() if len(l)>0]))
pdf.head()

sum_df = pd.DataFrame( df['single_text'] )
sum_df = sum_df.append(pd.DataFrame( pdf['single_text'] ))
sum_df.dropna(inplace=True)

