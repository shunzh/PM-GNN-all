import json

from TransformerModel.config import EXP_DIR
from TransformerModel.util import plot_hist

# EXP_DIR is the root the exp directory, e.g. EXP_DIR = '4comp_4000/'
INPUT = EXP_DIR + 'sim_analysis/analysis_result.txt'
EFF_OUTPUT = 'efficiency.json'
VOUT_OUTPUT = 'vout.json'


def parse():
    """
    Parse efficiency and v_out data in analysis_result.txt
    and save them in EFF_OUTPUT, VOUT_OUTPUT, respectively.

    Output formats:
    {'PCC-xxxxxx': efficiency, ...}
    {'PCC-xxxxxx': vout, ...}
    """
    file = open(INPUT, 'r')

    expect_eff = False
    expect_vout = False

    eff_dict = {}
    vout_dict = {}

    for line in file:
        if line.startswith('PCC'):
            id = line.strip()
            expect_eff = True
        elif expect_eff:
            if line.startswith('efficiency'):
                eff_str = line.strip().split(':')[1]
                # get rid of % symbol
                eff = int(eff_str.split('%')[0])
                eff_dict[id] = .01 * eff

                expect_vout = True

            expect_eff = False
        elif expect_vout:
            if line.startswith('output voltage'):
                vout_str = line.strip().split(':')[1]
                vout = int(vout_str)
                vout_dict[id] = vout

            expect_vout = False

    with open(EFF_OUTPUT, 'w') as f:
        json.dump(eff_dict, f)

    with open(VOUT_OUTPUT, 'w') as f:
        json.dump(vout_dict, f)

    # plot stats of eff and vout
    plot_hist(eff_dict.values(), 'Efficiency', 'eff', bins=50)
    plot_hist(vout_dict.values(), 'V_out', 'vout', bins=50)


if __name__ == '__main__':
    parse()
