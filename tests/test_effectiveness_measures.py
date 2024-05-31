from cana.boolean_node import BooleanNode
import math

measures = [
    "kr",
    "kr_norm",
    "ke",
    "ke_norm",
    "a",
    "s",
    "s_norm",
    # "kc",
    # "kc_norm"
]


def doTest(func, ans):
    k = int(math.log(len(func)) / math.log(2))
    node = BooleanNode(k=k, inputs=range(k), outputs=list(func))

    rets = {}
    for m in measures:
        if m == "kr":
            rets[m] = node.input_redundancy(norm=False)
        elif m == "kr_norm":
            rets[m] = node.input_redundancy(norm=True)
        elif m == "ke":
            rets[m] = node.effective_connectivity(norm=False)
        elif m == "ke_norm":
            rets[m] = node.effective_connectivity(norm=True)
        elif m == "a":
            rets[m] = node.activities()
        elif m == "s":
            rets[m] = node.sensitivity(norm=False)
        elif m == "s_norm":
            rets[m] = node.sensitivity(norm=True)
        # elif m == "kc":
        #     rets[m] = node.effective_connectivity(norm=False) - node.sensitivity(norm=False)
        # elif m == "kc_norm":
        #     rets[m] = node.effective_connectivity(norm=True) - node.sensitivity(norm=True)

    for m in measures:
        assert rets[m] == ans[m], m


def test_AND():
    f = "0001"
    ans = {
        "kr": 3 / 4,
        "kr_norm": 3 / 4 / 2,
        "ke": 5 / 4,
        "ke_norm": 5 / 4 / 2,
        "a": [0.5, 0.5],
        "s": 1,
        "s_norm": 1 / 2,
    }
    doTest(f, ans)
