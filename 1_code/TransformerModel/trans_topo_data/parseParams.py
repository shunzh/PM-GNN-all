import glob
import json

from TransformerModel.config import EXP_DIR, RANDOM_DATA

CKI_FILES = EXP_DIR + RANDOM_DATA + '*.cki'
OUTPUT = 'params.json'

def parse_params():
    """
    PARAM in cki file -> # params[PCC-xxxxxx]['key1'] = 'v1'
    """
    files = sorted(glob.glob(CKI_FILES))

    params = {}

    for filename in files:
        # get PCC-xxxxxx
        id = filename.strip().split('/')[-1][:-4]
        params[id] = {}

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('.PARAM'):
                    # ['key1=v1', 'key2=v2', ...]
                    param_and_values = line.strip().split(' ')[1:]

                    for param_and_value in param_and_values:
                        key, v = param_and_value.split('=')
                        # params[PCC-xxxxxx]['key1'] = 'v1'
                        params[id][key] = v

                    # only one line starts with PARAM, break here
                    break

    with open(OUTPUT, 'w')  as f:
        json.dump(params, f)


if __name__ == '__main__':
    parse_params()
