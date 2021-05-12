import os 

def getFileName(filename,path=os.getcwd(),outputfolder="outputs"):
    '''
    Returns a file name with a complete path attached to it. It can be configured for either windows or linux

    Side effects: if the directory path does not exist, then the directory will be created. 

    Inputs:
        filename, what the file should be called
        networkname, what the network is called this is used as the organizing directory in the outputs folder
        functionname, this is the organizational system within the networkname folder
    Output:
        filepathname, the complete file path to save off to. 
    Requires:
        import os
        
    '''
    if path=="": path=os.getcwd()
    filename = filename.replace(',','_').replace('/','_').replace('\\','_').replace('*','+')
    #Windows
    if os.name=='nt':
        outputfolder = outputfolder.replace('/','\\')
        filepath = '%s\\%s' % (path,outputfolder)
        filepathname = '%s\\%s' % (filepath,filename)
    #Linux
    else:
        outputfolder = outputfolder.replace('\\','/')
        filepath = '%s/%s' % (path,outputfolder)
        filepathname = '%s/%s' % (filepath,filename)
    if os.path.isdir(filepath) == False: {os.makedirs(filepath)}
    return filepathname