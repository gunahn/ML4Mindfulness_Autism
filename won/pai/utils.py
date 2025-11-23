import json
import pandas as pd
import textwrap

def convert_feature_name(orig_X: pd.DataFrame, txt_lim: str = 60):

    with open('./Questionnaire_names.json', 'r') as f:
        js = json.load(f)

    new_col = ['\n'.join(textwrap.wrap(c, txt_lim)) if js.get(c) is None else '\n'.join(textwrap.wrap(f"{js[c]}-{c}", txt_lim)) for c in orig_X.columns]
    orig_X.columns = new_col
    
    # return orig_X


if __name__ == '__main__':
    convert_feature_name(None)