import os
import sys
from ... boolean_network import BooleanNetwork

_path = os.path.dirname(os.path.realpath(__file__))

def getNeural(version,overlap):
    # get list of CNET files in the directory
    fileslst = os.listdir(_path)
    #choose the CNET that we are going to use
    for file in fileslst:
        if version in file and '.cnet' in file:
            if overlap==True:
                if "WOverlap" in file:
                    break
            else: 
                if "WoOverlap" in file:
                    break
    name = file.replace('.cnet','')
    #return it through processing the file
    return BooleanNetwork.from_file('%s\\%s'%(_path,file), name=name, filepath='%s\\%s'%(_path,file), keep_constants=True)