
import pandas as pd
from cana.networks.bio import MARQUESPITA
from cana.networks.bools import *
from cana.utils import *
from cana import bns
from cana import boolean_canalization as bc


#n = MARQUESPITA().nodes[6]
n = XOR()

print n.input_symmetry(mode='node', norm=False)



schema,perms,sms = ['201201'] , [[0,3,4,5]] , []
schema,perms,sms = ['201201'] , [[0,3,4,5]] , [[1,2]]

value = 0
# NEW VERSION
if len(perms+sms):
	value += sum( [ sum([len(x) for x in perms+sms if i in x]) for i in xrange(6)] ) / 6
print value

# Divide by the number of TS covering
value = value / len(self.ts_coverage[binstate])
symmetry.append(value)
k_s = sum(symmetry) / 2**6 # k_s
if (norm):
	k_s = k_s / 6

print k_s
"""
tdt0, tdt1 = bc.make_transition_density_tables(n.k, n.outputs)
print tdt0
print tdt1
print

pi0, pi1 = bc.find_implicants_qm(tdt0), bc.find_implicants_qm(tdt1)
print pi0
print pi1
print

print '-- V1'
#ts0s = bc.find_two_symbols_v1(n.k, pi0)
#ts1s = bc.find_two_symbols_v1(n.k, pi1)
#print ts0s
#print ts1s
print

print '-- V2'
ts0s = bc.find_two_symbols_v2(n.k, pi0, verbose=False)
ts1s = bc.find_two_symbols_v2(n.k, pi1)
print ts0s
print ts1s

n._check_compute_cannalization_variables(ts_coverage=True)
print n.ts_coverage

for input in xrange(n.k):
	for binstate, tss in n.ts_coverage.items():
		for schema,reps,sms in tss:
			print schema,reps,sms, '>', reps+sms
			for idxs in reps+sms:
				print idxs

ts_input_coverage = { input : { binstate: [ idxs.count(input) for schema,reps,sms in tss for idxs in reps+sms ] for binstate,tss in n.ts_coverage.items() } for input in xrange(n.k) }
print ts_input_coverage


for input,binstates in ts_input_coverage.items():
	print binstates.items()
	numstates = {binstate_to_statenum(binstate): permuts for binstate,permuts in binstates.items() }
	print numstates

"""

"""
pi_symbol=u'#'
ts_symbol=u"\u030A"
d = []
for output, ts in zip([0,1], [ts0s,ts1s]):
	for i,(schemata,perms,sms) in enumerate(ts,start=1):
		if len(perms):
			print schemata
			string = ''
			for j, perm in enumerate(perms,start=1):
				id = str(output) + '-' + str(i)
				if j>1:
					string += '|'
				string += ''.join([x if (k not in perm) else unicode(x)+ts_symbol for k,x in enumerate(schemata, start=0)])
				
			string = string.replace('2',pi_symbol)
			d.append( (id, string, output) )
		else:
			string = ''.join([x.replace('2',pi_symbol) for x in schemata])
			d.append( (string, output) )

print pd.DataFrame(d, columns=['id','In:','Out:'])

#dfLUT, dfPI, dfTW = n.look_up_table(), n.schemata_look_up_table(type='pi'), n.schemata_look_up_table(type='ts')
#print pd.concat({'Original LUT':dfLUT,'PI Schema':dfPI,'TW Schema':dfTW}, axis=1).fillna('-')

#print n.schemata_look_up_table(type='ts')
"""