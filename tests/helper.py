from itertools import permutations, product
from cana.datasets.bio import load_all_cell_collective_models
from cana.boolean_node import BooleanNode
import math
import random

def randNode(k):
    func = [random.randint(0,1) for i in range(2**k)]
    return BooleanNode(k=k, inputs=list(range(k)), outputs=func)

def reorderTwoSymbolOutput(tss):
    """Convert a list of two-symbol schemata to a set of two-symbol schemata with unique orderings for equality testing.
    
    arguments:
    tss -- two-symbol schemata list, [ (string, [[]], [[]]) ]
    """
    tssNew = set()
    for ts in tss:
        symGroups = ts[1]
        schemata = list(ts[0])
        for sg in symGroups:
            sgNew = sorted(sg)
            symsInSg = sorted([ (i, schemata[i]) for i in sgNew], key=lambda x: x[1])
            for i, sym in zip(sgNew, [i[1] for i in symsInSg]):
                schemata[i] = sym
        # tssNew.add( ("".join(schemata), frozenset(frozenset(i) for i in ts[1]), frozenset(frozenset(i) for i in ts[2])) )
        tssNew.add( ("".join(schemata), frozenset(frozenset(i) for i in ts[1]), frozenset()) ) # WARNING: ignoring same-symbol symmetry for now
    return tssNew

def expandTs(ts):
    """Expand a two-symbol schemata to the set of all schema (with don't cares) it encodes"""
    # expand ts
    tss = [i[0] for i in ts]
    perms = [i[1] for i in ts]
    obsSet = set()
    # for each schema and its symmetries
    for t, g in zip(tss, perms):
        if type(t) == str:
            t = list(t)
        # print(t, g)
        # for each subset of indices that can be permuted
        x = []
        for idxs in g:
            # print(idxs)
            # produce all permutations of those indices
            x.append([ (idxs, i) for i in permutations([t[j] for j in idxs],len(idxs))])
        # get cross-product of groups
        cxs = list(product(*x))
        # can apply each sequence in each item of cross product
        for seq in cxs:
            tPerm = t.copy()
            for p in seq:
                for i,j in zip(p[0], p[1]):
                    tPerm[i] = j
            # print(tPerm)
            obsSet.add("".join(tPerm))
    # return obsSet == trueSet, obsSet
    return obsSet

def enumerateImplicants(func):
    implicants = {"0": set(), "1": set()}
    k = int(math.log(len(func)) / math.log(2))
    for i, output in enumerate(func, start=0):
        cond = f"{bin(i)[2:]:0>{k}}"
        implicants[output].add(cond)
    return implicants

# each element of pi is a string
def expandPi(pi):
    """Expand a schemata with don't cares into the set of all schema it encodes"""
    out = set()
    for s in pi:
        # count number of 2s
        n = sum(1 for i in s if int(i) == 2)
        idxs = [i for i in range(len(s)) if s[i] == "2"]
        # get all permutations of 0 and 1 of that length
        # for perm in permutations("01", n):
        for perm in product(*["01"] * n):
            # print(perm)
            l = list(s)
            # produce substitution of each
            for k, i in enumerate(idxs):
                l[i] = perm[k]
            out.add("".join(l))
    return out

def compare(pi, ts):
    x = expandPi(expandTs(ts))
    y = expandPi(pi)
    return x==y, x-y, y-x

def getPis(outputs):
    """Compute prime implicants from a string function representation"""
    k = int(math.log(len(outputs)) / math.log(2))
    node = BooleanNode(k=k, inputs=range(k), outputs=list(outputs))
    node._check_compute_canalization_variables(prime_implicants="i dont matter")
    pi = node._prime_implicants
    return {0:set(i.replace("#", "2") for i in pi["0"]), 1:set(i.replace("#", "2") for i in pi["1"])}

def getCCnodes():
    networks = load_all_cell_collective_models()
    nodes = []
    for network in networks:
        for node in network.nodes:
            if node.k < 8 and "1" in node.outputs and "0" in node.outputs: # select non-constant with k<=7
                nodes.append(node)
    # fs = ["".join(n.outputs) for n in nodes]
    return nodes

if __name__ == "__main__":
    # l  = [ ("11", [], [[0,1]]) ]
    # x = reorderTwoSymbolOutput(l)
    # print(x)

    # l  = [ ("0110", [[0,1], [2,3]], []) ]
    # x = reorderTwoSymbolOutput(l)
    # print(x)

    # ts = [
    #         (list("0101"), [[1,2], [0,3]], []),
    #         (list("1201"), [[0,1]], [])
    #     ]
    # trueTs = set([
    #             "0101", "1010", "1100", "0011",
    #             "1201", "2101"
    #             ])
    # x = expandTs(ts)
    # # print(x)
    # print(x, trueTs == x)
    # print()
    # print(expandPi(trueTs))
    # print(expandPi(x))
    
    # t0 = [
    #         ("0202", [[0,1]], []),
    #         ("0220", [[0,1]], []),
    #         ("0202", [[2,3]], []),
    #         ("2002", [[2,3]], [])
    #      ]
    # print(expandTs(t0))
    # print(expandPi(expandTs(t0)))

    # x = {'221122', '222200', '002222', '202202', '022220'}
    # x = {"2212"}
    # print(expandPi(x))
    print(enumerateImplicants("01001011"))
