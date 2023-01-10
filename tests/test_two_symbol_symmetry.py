from helper import *
import math

from cana.canalization.boolean_canalization import * # WARNING: some functions here differ from the file below!
from cana.canalization.cboolean_canalization import *
from cana.boolean_node import BooleanNode
from cana.cutils import outputs_to_binstates_of_given_type

# WARNING: ignoring detection of same-symbol symmetry for now. Complicating issues.

def getTss(outputs):
    """Compute unique representation of two-symbol schemata from Boolean output table.
    Assume that prime-implicant caclulation is correct.
    
    arguments:
    outputs -- string representing the lookup table of the function
    """
    k = int(math.log(len(outputs)) / math.log(2))
    # bs0 = outputs_to_binstates_of_given_type(outputs=outputs, output=0, k=k)
    # bs1 = outputs_to_binstates_of_given_type(outputs=outputs, output=1, k=k)
    # pi0, pi1 = find_implicants_qm(bs0), find_implicants_qm(bs1) # TODO: weird issue. when this file run as script causes error. 
    # pi0 = set([pi.replace('#', '2') for pi in pi0])
    # pi1 = set([pi.replace('#', '2') for pi in pi1])
    # print(pi0)
    # ts0, ts1 = find_two_symbols_v2(k=k, prime_implicants=pi0), find_two_symbols_v2(k=k, prime_implicants=pi1)
    
    # ALTERNATIVE
    node = BooleanNode(k=k, inputs=range(k), outputs=list(outputs))
    node._check_compute_canalization_variables(two_symbols="i dont matter")
    ts0, ts1 = node._two_symbols

    return {0:ts0, 1:ts1}

def test_bulk():
    nodes = getCCnodes()#[0:800]
    fails = []
    log = []
    for node in nodes:
        f = node.outputs
        ts = getTss(f)
        pi = getPis(f)
        for y in [0,1]:
            x = compare(pi[y], ts[y])
            if not x[0]: # False if not same sets
                fails.append(x)
                log.append( ", ".join([node.network.name, node.name, "".join(str(i) for i in node.outputs), str(y), str(node.k)] ))
    # print(len(fails), fails)
    # print([i for i in fails if i[2] != set()])
    with open("bulk_test_log.csv", "w") as fd:
        columns = ["network", "node", "function", "PIs", "k"]
        fd.write(", ".join(columns))
        fd.write("\n")
        fd.write("\n".join(log))
    assert len(fails) == 0, f"{len(fails)} failures"

def doTSStest(func, true_ts0, true_ts1):
    true_ts0 = reorderTwoSymbolOutput(true_ts0)
    true_ts1 = reorderTwoSymbolOutput(true_ts1)
    tsss = getTss(func)

    assert (reorderTwoSymbolOutput(tsss[0]) == true_ts0), (tsss[0])
    assert (reorderTwoSymbolOutput(tsss[1]) == true_ts1), (tsss[1])

def test_two_symbol_AND():
    f = "0001"
    true_ts0 = [("20", [[0,1]], [])]
    true_ts1 = [("11", [], [[0,1]])]
    doTSStest(f, true_ts0, true_ts1)

def test_two_symbol_OR():
    f = "0111"
    true_ts0 = [("00", [], [])]
    true_ts1 = [("12", [[0,1]], [])]
    doTSStest(f, true_ts0, true_ts1)

# TODO: randomly fails test sometimes. sometimes returns t0 with [0,1] same-symbol group
def test_two_symbol_AB_C():
    f = "01010111"
    t0 = [
            ("020", [[0,1]], [])
         ]
    t1 = [
            ("112", [], []),
            ("221", [], [])
         ]
    doTSStest(f, t0, t1)

def test_two_symbol_ABC():
    f = "00000001"
    t0 = [
            ("022", [[0,1,2]], [])
         ]
    t1 = [
            ("111", [], [])
         ]
    doTSStest(f, t0, t1)

def test_two_symbol_AB_CD_lv1():
    f = "0001000100011111"
    t0 = [
            ("0202", [[0,1]], []),
            ("0220", [[0,1]], []),
            ("0202", [[2,3]], []),
            ("2002", [[2,3]], [])
         ]
    t1 = [
            ("1122", [], []),
            ("2211", [], [])
         ]
    doTSStest(f, t0, t1)

# NOTE: this is a test for when it can detect two groups at once in this case
# def test_two_symbol_AB_CD_lv2():
#     f = "0001000100011111"
#     t0 = [
#             ("0202", [[0,1],[2,3]], []), # WARNING should this one have same-symbol permutations?
#          ]
#     t1 = [
#             ("1122", [], []),
#             ("2211", [], [])
#          ]
#     doTSStest(f, t0, t1)

def test_two_symbol_AP3():
    f = "00000000000000000000000000000000000000001111111100000000111111110000000100000001000100010001000100000001111111110001000111111111"
    t0 = [
            ("2002022", [[1,3]], []),
            ("2022220", [[0,5,6],[1,3]], [])
         ]
    t1 = [
            ("2121222", [], []),
            ("1222111", [[2,4]], [])
         ]
    doTSStest(f, t0, t1)

def test_symbol_LFY():
    f = "1110111011101110"
    t0 = [
            ("2211", [], [])
         ]
    t1 = [
            ("2202", [[2,3]], [])
         ]
    doTSStest(f, t0, t1)

def test_two_symbol_Thaliana_AG():
    f = "11001100110011001100110011001100110011001100110011001100110011000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111100110011001100110011001100110011001100110011001100110011001100000000000000000000000000000000000000000000000000000000000000000011111111111111111111111111111111111111111111111111111111111111111111111111110000111111111111010111111111111111111111111111111111"
    t0 = [
            ("121001122", [[4,8]], []),
            ("201222222", [[2,7]], [])
         ]
    t1 = [
            ("220222202", [], []),
            ("212212221", [], []),
            ("212122222", [], []),
            ("012222222", [[0,2,5,6]], [])
         ]
    doTSStest(f, t0, t1)


def test_two_symbol_BuddingYeast_Cdh1():
    f = "01001101000001000000010000000000"
    t0 = [
            ("22020", [], []),
            ("21212", [[0,1,3]], []),
            ("21220", [[0,1,3],[2,4]], [])
         ]
    t1 = [
            ("00201", [[2,4]], []),
            ("00121", [[0,1,3]], [])
         ]
    doTSStest(f, t0, t1)

def test_two_symbol_Lymphoid_IL7r():
    f = "0000000000000000010101010100010000001111000011000101111101001100"
    t0 = [
            ("221212", [], []),
            ("002222", [[1,5]], []),
            ("222020", [[0,3]], []),
            ("202022", [[0,3]], []),
            ("222020", [[1,5]], [])
         ]
    t1 = [
            ("010221", [[2,4]], []),
            ("120122", [[2,4]], [])
         ]
    doTSStest(f, t0, t1)

if __name__ == "__main__":
    # f = "00000001"
    # ts = getTss(f)
    # # x = computes_ts_coverage(3, f, ts[0])
    # t0 = [
    #         (list("022"), [[0,1,2]], [])
    #      ]
    # t1 = [
    #         (list("111"), [], [])
    #      ]

    # x = compareTs(ts[0], t0)
    # print(x)
    # x = compareTs(ts[1], t1)
    # print(x)

    # f = "0001000100011111"
    # t0 = [
    #         ("0202", [[0,1]], []),
    #         ("0220", [[0,1]], []),
    #         ("0202", [[2,3]], []),
    #         ("2002", [[2,3]], [])
    #      ]
    # t1 = [
    #         ("1122", [], []),
    #         ("2211", [], [])
    #      ]
    # ts = getTss(f)
    # print(ts)
    # x = compareTs(ts[0], t0)
    # print(x)
    # x = compareTs(ts[1], t1)
    # print(ts[1])
    # print(x)
    
    f = "0000000000000000010101010101000000110011001100000111011101110000"
    pis = getPis(f)
    print(pis[0])
    ts = getTss(f)
    print(ts[0])
    print(compare(pis[0], ts[0]))