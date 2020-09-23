from cana.canalization import boolean_canalization as BCanalization
from cana.datasets.bio import THALIANA
from cana.datasets.bools import AND


n = AND()
print(n)
#print(n.nodes[0])
print('inputs:',n.inputs)
print('mask:',n.mask)

n._check_compute_canalization_variables(prime_implicants=True)
pi = n._prime_implicants
print(type(pi), pi)

pi0 = pi['0']
pi1 = pi['1']


print('-- PI: --')
print(pi0)
print(pi1)

pi0 = set([pi.replace('#', '2') for pi in pi0])
pi1 = set([pi.replace('#', '2') for pi in pi1])

ts0 = BCanalization.find_two_symbols_v2(k=n.k, prime_implicants=pi0)
ts1 = BCanalization.find_two_symbols_v2(k=n.k, prime_implicants=pi1)

print('-- TS: --')
print(ts0)
print(ts1)

#n._check_compute_canalization_variables(two_symbols=True)
#print(n._two_symbols)


print ('-- Canalizing Map --')
#n = THALIANA()

G = n.canalizing_map()

print(G.nodes(data=True))