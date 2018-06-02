import os,re
from itertools import product

def CC_bool_expr2cnet(expression_all_file,external_node_file,outputfile):
    '''Convert ONE network in Cell Collective Boolean expression format to a cnet file. 

    Args:
        expression_all_file (string) : the location of the file containing Boolean expressions
        external_node_file (string) : the location of the file containing external nodes list
        outputfile (string) : the location of the cnet file that you want to save

    Returns:
        None

    Example: 
        CC_bool_expr2cnet('Lac Operon/expressions.ALL.txt','Lac Operon/external_components.ALL.txt','Lac Operon.cnet')

    See also:
        :attr:'CC_bool_expr_folder_to_cnet', :attr:'batch_CC_bool_expr_to_cnet'
    '''
    with open(outputfile,'w') as output:

        outputlines = []
        outputhead = []
        
        NodeList = []
        with open(expression_all_file) as expfile:
            for line in expfile:
                # search for "NodeName = expression" and return the NodeName
                rem = re.match(r'^(\S+)\s*=\s*(.+)$',line)
                if rem is None:
                    print('wrong content!')
                    return
                NodeList.append(rem.group(1))                
        
        NodeDict = {}
        # case sensitive.  Hope Cell Collective data contains no bug.
        # if there is case insensitive data, we should add upper() to this and
        # add upper() to nodefile

        for i in range(len(NodeList)):
            NodeDict[NodeList[i]] = i + 1

        # any node beyond this mark will be external node
        externalp = len(NodeList)


        # add external nodes.  Nodes without input
        # edit: no longer use old methods since Cell Collective has provided
        # external nodes!
        with open(external_node_file) as f:
            for line in f:
                word = line.strip()
                NodeList.append(word)
                NodeDict[word] = len(NodeList)

        # initiate node value list
        # comparing with substituting truth value, value assignment shoud be
        # much faster!
        # string search and replace is slow
        NodeValueList = [None] * len(NodeList)


        # write nodes number
        outputhead.append('.v %s\n' % len(NodeList))
        outputhead.append('\n')

        
        # write node list
        for i in range(len(NodeList)):
            outputhead.append('.l %s %s\n' % (i + 1,NodeList[i]))      

        outputhead.append('\n')
        output.writelines(outputhead)

        # begin construct Boolean Function
        FunctionDict = {}
        with open(expression_all_file) as expfile:
            for line in expfile:
                line = line.strip()
                rem = re.match(r'^(\S+)\s*=\s*(.+)$',line)                
                FunctionDict[rem.group(1)] = _parse_bool_func(rem.group(2),NodeDict)

        
        for node in NodeList[:externalp]:
            outputlines.append('# %s %s \n' % (NodeDict[node],node))
            # this is the format required by cnet
            # node number, input number, then all input node number
            namelabel = [NodeDict[node],len(FunctionDict[node][0])] + [NodeDict[i] for i in FunctionDict[node][0]]
            outputlines.append('.n ' + ' '.join([str(i) for i in namelabel]) + '\n')
            # abandon writing all lines at once in the end.  Because some LUTs
            # are huge
            output.writelines(outputlines)
            outputlines = []
            # enumerate all possible combination of truth values
            for input in product([True,False],repeat=len(FunctionDict[node][0])):
                for i in range(len(FunctionDict[node][0])):
                    NodeValueList[NodeDict[FunctionDict[node][0][i]] - 1] = input[i]
                try:
                    result = eval(FunctionDict[node][1])
                except Exception as e:
                    print('Error in evaluating Boolean expression!')
                    print(e)
                    print(FunctionDict[node][1])
                    break
                if result is None:
                    print('Error in evaluating Boolean expression!')
                    print(FunctionDict[node][1])
                    break
                line = _LUTrow(input,result)
                output.write(line)
            output.write('\n')
            


        for i in range(externalp,len(NodeList)):
            outputlines.append('# %s %s \n' % (i + 1,NodeList[i]))
            outputlines.append('.n %i 0\n' % (i + 1))
            outputlines.append('\n')
        outputlines.append('.e End of file\n')
        output.writelines(outputlines)


def _LUTrow(assignment,result):
    ''' interal function. Given value assignment of input nodes, return the LUT line needed for cnet format

    Args:
        assignment (iterable) : a list containing truth values 
        result (bool) : the result of the expression truth value upon the assignment

    Returns:
        string : the row that can be written to cnet file

    Example:
        ([True,False], True) -> '10 1\n'
    '''
    return ''.join(['1' if i else '0' for i in assignment]) + ' ' + ('1' if result else '0') + '\n'

def _parse_bool_func(exp,NodeDict):
    '''interal function. convert a Boolean expression to an expression can be evaluated by python
    Don't use this outside this module. It won't have any meaning unless calling from CC_bool_expr2cnet()!

    Args:
        exp (string) : Boolean expressions extracted from Cell Collective Boolean expression files
        NodeDict (dict) : a dict containing (node name)->(node number) map

    Returns:
        list : a list containing all node names occur in exp
        string : a converted string that can be evaluated by python interpreter

    '''
    # abandon old way.  Use more secure and general way to extract node names
    # node names must not contain = and spaces !
    # update, now can process node names containing ()
    # node names must not be AND NOT OR !
    # update: should consider node names that are subset of other node names
    varlist = []
    NodeList_sorted = [nodename for nodename in NodeDict]
    NodeList_sorted.sort(key=lambda x:len(x),reverse=True)
    exp_copy = exp
    for nodename in NodeList_sorted:
        if nodename in exp_copy:
            varlist.append(nodename)
            exp_copy = exp_copy.replace(nodename,'')
    sublist = [(varlist[i],'NodeValueList[' + str(NodeDict[varlist[i]] - 1) + ']') for i in range(len(varlist))]
    map_lower_case = [('AND','and'), ('OR','or'), ('NOT','not')]
    # even keywords should be sorted in the same pool
    # or node names like 'AN' 'NO' would cause errors
    sublist+=map_lower_case
    # it is mandatory to substitute the longer string first
    # Sometime a shorter string is a part of a longer string
    sublist.sort(key=lambda x:len(x[0]),reverse = True)
    newexp = exp
    for i,j in sublist:
        newexp = newexp.replace(i,j)
    return varlist,newexp


def CC_bool_expr_folder_to_cnet(expfolder,outputfile):
    '''give a folder containing Cell Collective Boolean expression files, convert to a cnet file

    Args:
        expfolder (string) : the location of folder containing expressions.ALL.txt and external_components.ALL.txt
        outputfile (string) : the location of the cnet file that you want to save

    Returns:
        None

    Example:
        CC_bool_expr_folder_to_cnet('./Lac Operon','Lac Operon.cnet')

    See also:
        :attr:'CC_bool_expr2cnet', :attr:'batch_CC_bool_expr_to_cnet'
    '''
    CC_bool_expr2cnet(os.path.join(expfolder,'expressions.ALL.txt'),os.path.join(expfolder,'external_components.ALL.txt'),outputfile)


def batch_CC_bool_expr_to_cnet(inputfolder,outputfolder):
    '''Batch convert all networks in a folder to cnet files
    This function will recursively visit all subfolders and convert all networks it finds. 
    It is useful when you extract all zip files you download from Cell Collective and want to convernt them all

    Args:
        inputfolder (string) : location of the folder containing your input files
        outputfolder (string) : location that you want to save those converted cnet files

    Returns:
        None

    See also:
        :attr:'CC_bool_expr_folder_to_cnet'
    '''
    for root,folders,files in os.walk(inputfolder):
        if len(files) > 0 and os.path.splitext(files[0])[1] == '.txt':
            if os.path.split(root)[-1] == 'expr':
                Networkname = root.split(os.path.sep)[-2]
            else:
                Networkname = os.path.split(root)[-1]
            print('Processing ' + Networkname)
            try:
                CC_bool_expr_folder_to_cnet(root,os.path.join(outputfolder,Networkname + '.txt'))
            except Exception as e:
                print(e)
                print(root)
                #print('input file %s in %s contains error, could not convert'
                #% (e[0],e[1]))
                #if os.path.exists(e[2]):
                #    os.remove(e[2])
                #    print('%s is deleted' % e[2])
                #pass

