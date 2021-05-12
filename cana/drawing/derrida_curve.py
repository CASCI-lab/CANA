import matplotlib.pyplot as pyplot
from cana.drawing.save_utils import getFileName

def derrida_curve(derridaCurve,
    SaveFolder="", # suggested file path is "\\outputs\\<networkname>"
    SaveName="", # if none is given it will be derridaCurve.png, suggest to append network name to it. 
    SaveFormat="png",
    path="", #if no path is given path is os.getcwd()
    OpenGraph=False,
    OneOneLine=True,
    DotColor="ro"
    ):
    pyplot.plot(derridaCurve[0],derridaCurve[1],DotColor)
    if OneOneLine: pyplot.plot((0,1),(0,1),'--')
    if SaveFolder!="":
        filename = getFileName("%s.%s"%(SaveName,SaveFormat),path=path,outputfolder=SaveFolder)
        pyplot.savefig(filename,format=SaveFormat)
    if OpenGraph: pyplot.show() 