import shutil
import warnings
from lcapy import Circuit
import json

from gen_topo_for_analytic import *
from utils import *


def key_circuit_from_lists(edge_list, node_list, net_list):
    path = find_paths_from_edges(node_list, edge_list)

    node_dic = {}
    node_name = {}
    net_list_dic = {}

    for edge in edge_list:

        edge_start = edge[0]
        edge_end = edge[1]

        if edge_end in node_dic:
            node_dic[edge_end].append(edge_start)
        else:
            node_dic[edge_end] = []
            node_dic[edge_end].append(edge_start)

    for node in node_dic:

        node_dic[node].sort()
        name = 'N'

        for comp in node_dic[node]:
            name = name + '-' + comp

        if node in node_name:
            print('error')
        else:
            node_name[str(node)] = name

    tmp = net_list
    for item in tmp:

        for index, node in enumerate(item):
            if node == '0':
                item[index] = '0'
            elif node in node_name:
                item[index] = node_name[node]
        net_list_dic[item[0]] = item[1::]
        net_list_dic[item[0]].sort()

    net_list_dic_sorted = OrderedDict(sorted(net_list_dic.items()))

    key = str(net_list_dic_sorted)

    return key


def get_one_circuit(key, net_list, parameters):
    device_list = []
    node_list = []
    param = parameters
    # circuit_a = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 100\n' + 'Cout OUT 0\n'
    # circuit_b = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 100\n' + 'Cout OUT 0\n'
    circuit_a = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 50\n' + 'Cout OUT 0\n'
    circuit_b = 'Vin IN_exact 0 100\nRin IN_exact IN 0.1\n' + 'Rout OUT 0 50\n' + 'Cout OUT 0\n'

    device_list = ['Vin', 'Rin', 'Rout', 'Cout']

    node_list = ['IN', 'OUT', 'IN_exact']

    for item in net_list:
        line_a = ''
        line_b = ''
        device = item[0]

        if device[0:2] == 'Sa':
            line_a = 'Ra' + device[2:]
            line_b = 'Rb' + device[2:]
            if line_a not in device_list:
                device_list.append(line_a)
            if line_b not in device_list:
                device_list.append(line_b)
        elif device[0:2] == 'Sb':
            line_a = 'Rb' + device[2:]
            line_b = 'Ra' + device[2:]
            if line_a not in device_list:
                device_list.append(line_a)
            if line_b not in device_list:
                device_list.append(line_b)
        else:
            line_a = device
            device_list.append(line_a)
            line_b = device

        for node in item[1::]:
            line_a = line_a + ' ' + node
            line_b = line_b + ' ' + node
            if node not in node_list:
                node_list.append(node)
        if device[0] == 'C':
            line_a = line_a + ' ' + str(param['C'][0])
            line_b = line_b + ' ' + str(param['C'][0])
        if device[0] == 'L':
            line_a = line_a + ' ' + str(param['L'][0])
            line_b = line_b + ' ' + str(param['L'][0])
        """Possible cases"""
        if device[0:2] == 'Sa':
            line_a = line_a + ' ' + str(param['Ra'][0])
            line_b = line_b + ' ' + str(param['Rb'][0])
        if device[0:2] == 'Sb':
            line_a = line_a + ' ' + str(param['Rb'][0])
            line_b = line_b + ' ' + str(param['Ra'][0])

        line_a = line_a + '\n'
        line_b = line_b + '\n'
        circuit_a = circuit_a + line_a
        circuit_b = circuit_b + line_b

    circuit_info = {"key": key, "circuit_a": circuit_a, "circuit_b": circuit_b, "device_list": device_list,
                    "node_list": node_list, "net_list": net_list}

    return circuit_info


def get_one_expression(circuit_info):
    """

    @param circuit_info:
    @return: expression
    """
    invalid_topo = []

    data = {}

    circuit_a = circuit_info['circuit_a'].split('\n')
    circuit_b = circuit_info['circuit_b'].split('\n')
    device_list = circuit_info['device_list']
    node_list = circuit_info['node_list']
    net_list = circuit_info['net_list']

    cct_a = Circuit()
    cct_b = Circuit()

    for item in circuit_a:
        #            if item !='' and item[0]=='V':
        cct_a.add(item)

    for item in circuit_b:
        #            if item !='' and item[0]=='V':
        cct_b.add(item)

    try:
        ss_a = cct_a.ss
        ss_b = cct_b.ss
    except:
        invalid_topo.append(circuit_info['key'])
        print("%s violations circuit\n")
        return "invalid"

    a_X = str(ss_a.x)[7:-1]
    a_Y = str(ss_a.y)[7:-1]
    a_A = str(ss_a.A)[7:-1]
    a_B = str(ss_a.B)[7:-1]
    a_C = str(ss_a.C)[7:-1]
    a_D = str(ss_a.D)[7:-1]

    a = {'x': a_X,
         'y': a_Y,
         'a': a_A,
         'b': a_B,
         'c': a_C,
         'd': a_D
         }

    b_X = str(ss_b.x)[7:-1]
    b_Y = str(ss_b.y)[7:-1]
    b_A = str(ss_b.A)[7:-1]
    b_B = str(ss_b.B)[7:-1]
    b_C = str(ss_b.C)[7:-1]
    b_D = str(ss_b.D)[7:-1]

    b = {'x': b_X,
         'y': b_Y,
         'a': b_A,
         'b': b_B,
         'c': b_C,
         'd': b_D
         }

    expression = {'A state': a, 'B state': b, 'device_list': device_list, 'node_list': node_list,
                  'net_list': net_list}
    return expression


def get_analytics_result(parameters, expression):
    """

    @param parameters:
    @param expression:
    @return: [fn, duty_cycle, str(vect), vout, effi, flag_candidate, name_list, net_list]
    """
    assert expression
    fn = 'one_circuit'
    net_list = expression['net_list']

    a_x = expression['A state']['x']
    a_y = expression['A state']['y']
    a_A = expression['A state']['a']
    a_B = expression['A state']['b']
    a_C = expression['A state']['c']
    a_D = expression['A state']['d']

    b_x = expression['B state']['x']
    b_y = expression['B state']['y']
    b_A = expression['B state']['a']
    b_B = expression['B state']['b']
    b_C = expression['B state']['c']
    b_D = expression['B state']['d']

    device_list = expression['device_list']

    param2sweep, paramname = gen_param(device_list, parameters)

    paramall = list(it.product(*(param2sweep[Name] for Name in paramname)))

    name_list = {}
    for index, name in enumerate(paramname):
        name_list[name] = index
    vect = paramall[0]
    # for vect in paramall:
    # print(fn, vect)
    duty_cycle = vect[name_list['Duty_Cycle']]
    vin = vect[name_list['Vin']]
    rin = vect[name_list['Rin']]
    rout = vect[name_list['Rout']]
    freq = vect[name_list['Frequency']]
    cout = vect[name_list['Cout']]

    a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = a_x, a_y, a_A, a_B, a_C, a_D
    b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = b_x, b_y, b_A, b_B, b_C, b_D

    nodelist = a_y[2:-2].split('], [')
    statelist = a_x[2:-2].split('], [')

    k = 0
    for node in nodelist:

        if str(node) == 'v_IN(t)':
            Ind_Vin = k
            flag_IN = 1
        if str(node) == 'v_IN_exact(t)':
            Ind_Vinext = k
            flag_IN_ext = 1
        if str(node) == 'v_OUT(t)':
            Ind_Vout = k
            flag_OUT = 1
        k = k + 1

    for index, value in enumerate(vect):
        a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp = exp_subs(a_xp, a_yp, a_Ap, a_Bp, a_Cp, a_Dp, paramname[index],
                                                      str(value))
        b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp = exp_subs(b_xp, b_yp, b_Ap, b_Bp, b_Cp, b_Dp, paramname[index],
                                                      str(value))

    A = duty_cycle * np.array(eval(a_Ap)) + (1 - duty_cycle) * np.array(eval(b_Ap))
    B = duty_cycle * np.array(eval(a_Bp)) + (1 - duty_cycle) * np.array(eval(b_Bp))
    C = duty_cycle * np.array(eval(a_Cp)) + (1 - duty_cycle) * np.array(eval(b_Cp))
    D = duty_cycle * np.array(eval(a_Dp)) + (1 - duty_cycle) * np.array(eval(b_Dp))
    try:
        A_inv = np.linalg.inv(A)
    except:
        return None

    x_static = -np.matmul(A_inv, B) * vin
    y_static = (-np.matmul(np.matmul(C, A_inv), B) + D) * vin

    Vout = y_static[Ind_Vout]
    Iin = abs((y_static[Ind_Vin] - y_static[Ind_Vinext]) / rin)
    Pin = Iin * vin
    Pout = Vout * Vout / rout
    eff = Pout / (Pin + 0.01)

    vout = int(Vout[0])
    effi = float(int(eff[0] * 100)) / 100
    flag_candidate = (vout > -500) and (vout < 500) and (vout < vin * 0.6 or vout > vin * 1.2) and 60 < effi < 100

    effi_vout_info = [fn, duty_cycle, str(vect), vout, effi, flag_candidate, name_list, net_list]

    return effi_vout_info


def get_prediction_from_topo_info(list_of_node, list_of_edge, net_list, param, target_vout=50, expression=None):
    """
    get the prediction of a topology with list information and parameter
    @param expression: if we have the expression, we can directly use it
    @param param: the [duty cycle, C, L] format, need to be add into a full format
    @param target_vout: target output voltage
    @param list_of_node:
    @param list_of_edge:
    @param net_list:
    @return:efficiency, vout, reward
    """
    parameters = json.load(open("./param.json"))
    fix_paras = {'Duty_Cycle': [param[0]], 'C': [param[1]], 'L': [param[2]]}
    parameters = assign_DC_C_and_L_in_param(param=parameters, fix_paras=fix_paras)
    effi = -1, vout = -500, reward = 0

    if not expression:
        # generate the expression using topo information
        key = key_circuit_from_lists(edge_list=list_of_edge, node_list=list_of_node, net_list=net_list)
        circuit_info = get_one_circuit(key=key, net_list=net_list, parameters=parameters)
        expression = get_one_expression(circuit_info=circuit_info)
    # effi_vout_info: [fn, duty_cycle, str(vect), vout, effi, flag_candidate, name_list, net_list]
    effi_vout_info = get_analytics_result(expression=expression, parameters=parameters)

    if effi_vout_info:
        effi = effi_vout_info[4], vout = effi_vout_info[3]
        reward = calculate_reward(effi={'efficiency': effi, 'output_voltage': vout},
                                  target_vout=target_vout)
    return effi, vout, reward, expression
