import random
from cana.boolean_network import BooleanNetwork
from cana.drawing.save_utils import getFileName
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

'''
Requires:
    import random (for the graphic)
Input:
    Comparisons. Following are common comparisons
        ('k_r*','k_s*'),('k','k_r*'),('k','k_s*'),('k','k_r')
        Possible axes:
            'k','k_r','k_e','k_s','k_r*','k_e*','k_s*','k_e(out-d)','k_e(out-s)'
'''
def NodeMeasures(N,Neg,
                path='',
                comparisons=[('k_r*','k')],
                fileformat='png',
                outputfolder=''
                ):
    if outputfolder=='': outputfolder='outputs\\%s'%(N.name)
    used = []
    used.extend([i[0] for i in comparisons])
    used.extend([i[1] for i in comparisons])
    used = list(set(used))
    if Neg is None and (('k_e(out-d)' in used) or ('k_e(out-s)' in used)):
        threshold = 0.00
        Neg = N.effective_graph(mode='input',bound='upper', threshold=threshold)
    bound = 'upper'
    #ohly calculate what you need
    dicf = {'node':[n.name for n in N.nodes],'k':[n.k for n in N.nodes]}
    dflabel=['k']
    for i in used:
        if i =='k_r':
            dicf['k_r'] = [n.input_redundancy(mode='node',bound=bound,norm=False) for n in N.nodes]
            dflabel.append('k_r')
        elif i =='k_e':
            dicf['k_e'] = [n.effective_connectivity(mode='node',bound=bound,norm=False) for n in N.nodes]
            dflabel.append('k_e')
        elif  i =='k_s':
            dicf['k_s'] = [n.input_symmetry(mode='node',bound=bound,norm=False) for n in N.nodes]
            dflabel.append('k_s')
        elif  i =='k_r*':
            dicf['k_r*'] = [n.input_redundancy(mode='node',bound=bound,norm=True) for n in N.nodes]
            dflabel.append('k_r*')
        elif  i =='k_e*':
            dicf['k_e*'] = [n.effective_connectivity(mode='node',bound=bound,norm=True) for n in N.nodes]
            dflabel.append('k_e*')
        elif i =='k_s*':
            dicf['k_s*'] = [n.input_symmetry(mode='node',bound=bound,norm=True) for n in N.nodes]
            dflabel.append('k_s*')
        elif i =='k_e(out-d)':
            dicf['k_e(out-d)'] = [Neg.out_degree()[n] for n in Neg.out_degree()]
            dflabel.append('k_e(out-d)')
        elif i =='k_e(out-s)':
            dicf['k_e(out-s)'] = [Neg.out_degree(weight='weight')[n] for n in Neg.out_degree(weight='weight')]
            dflabel.append('k_e(out(s)')
    df = pd.DataFrame(dicf).set_index('node')
    df = df[dflabel]
    # efile = GetFileName('%s_NodeMeasures.csv'%N.name,"NeuralReducedFigures","NodeMeasures") # ('NodeMeasures.csv',N.name,"NodeMeasures")
    # df.to_csv(efile, encoding='utf-8')
    
    for x,y in comparisons:
        fig, ax = plt.subplots(1,1,figsize=(6,5), sharex=True, sharey=True)
        dfp = df.loc[ (df['k']>1) , :]
        ax.scatter(dfp[x],dfp[y], s=50, c='red', marker='o', zorder=2)
        # lx,ly = [],[]
        # quadrants = [-0.035,0.035]
        # for name, dfp_ in dfp.iterrows():
        #     x,y = dfp_[x]+random.choice(quadrants) , dfp_[y]+random.choice(quadrants)
        #     ax.annotate(name, (x,y),fontsize=12, va='center', ha='center')
        #     lx.append(x); ly.append(y)
        xmax= np.ceil(max(df[x]))
        ymax= np.ceil(max(df[y]))
        ax.plot((0,xmax),(0,ymax),'black', lw=2,alpha=0.25, zorder=1)
        ax.grid(True)
        ax.set_xlim(-0.05,xmax+0.05)
        ax.set_ylim(-0.05,ymax+0.05)
        ax.set_xlabel('$%s$'%x)
        ax.set_ylabel('$%s$'%y)
        filename = getFileName('%s~%s~%s~Graph.%s'%(N.name,x,y,fileformat),path=path,outputfolder=outputfolder)
        plt.savefig(filename, dpi=150,format=fileformat)
        plt.clf()
        plt.close('all')