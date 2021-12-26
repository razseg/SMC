# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:19:15 2021

@author: Segal Raz
"""



import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import time
import os
import string
import copy
import InfoCom_functions as BNalg
import LoadAlgNew as Lalg
import LoadOnEdgesAlg as Walg
import ast
import random
import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def paseDistrebution(file):
    distributionFile=open(file)
    distRead=distributionFile.read()
    distRead=distRead.split('\n')
    List=[]
    for line in distRead:
        line=line.replace("{", "")
        line=line.replace("}", "")
        line=line.replace(" ", "")
        line=line.split(":")
        List.append((line[0],line[1].split(",")))
    return List
def LevelColor(level):
    return [x for x in range(2**level-1,2**(level+1)-1)]

def TopColor(amount,g,leafList):
    deg=[]
    for leaf in leafList:
        deg.append(g.nodes[leaf]['load'])
    ret=[x for x in range(0,amount-1)]
    ret.append(deg.index(max(deg)))
    return ret

def MaxColor(amount,g,leafList):
    deg=[]
    for leaf in leafList:
        deg.append(g.nodes[leaf]['load'])
    l=[]
    for i in range(0,amount):
        l.append(leafList[deg.index(max(deg))])
        deg[deg.index(max(deg))]=0
    return l
def leafList(g):
    return [x for x in g.nodes() if g.out_degree(x)==0 and g.in_degree(x)==1]
def addLoad(g,load,nodesList):
    for n in g.nodes:
        if n in nodesList:
            att={ n : {'load':int(load[nodesList.index(n)])}}
        else:
            att={ n : {'load':0}}
        nx.set_node_attributes(g,att)
             
def GtoFile(g,name):
    f=open(name+".txt","w")
    for node in g.nodes():
        f.write('node: '+str(node)+'\n')
        for i in g.nodes[1]['minSend']:
                f.write(str(i)+ ' U: '+str(g.nodes[node]['minSend'][i]['U'])+', bn: '+str(g.nodes[node]['minSend'][i]['bn'])+'\n')
    f.close()

def colorMap(gr):
        color_map = []
        for node in gr.nodes:
            color_map.append(gr.nodes[node]['color'])
        return color_map

def maxBottleNeck_messageNum(g):
    tmp=[]
    for e in g.edges:
        tmp.append(g.edges[e]['mesageCount'])
    return [np.sum(tmp),max(tmp)]

def NetworkUtiliztion(g):
    tmp=[]
    for e in g.edges:
        tmp.append(g.edges[e]['mesageCount']*(1/g.edges[e]['Wieght']))
    return [np.sum(tmp),max(tmp)]    
    
def sumLoad(g):
    LoadSum=0
    for node in g.nodes:
        LoadSum=LoadSum+ g.nodes[node]['load']
    return LoadSum
def minMaxLink(g):
    arr=[]
    for e in g.edges:
        arr.append(g.edges[e]["Wieght"])
    return [min(arr),max(arr)]
def findX(g,root,k,Avilabilty):
    vec=SerchVector(g,root)
    L=0
    R=len(vec)
    
    while L <= R and  (R-L)>= 1 :
        # print('L: '+str(L)+' R: '+str(R)+' X: '+str(vec[int((L+R)/2)]))
        BNalg.run(g,root,k,(vec[int((L+R)/2)]),Avilabilty)
        if g.nodes[0]['minSend']['k'+str(k)]['bn']:
            # if R != ((L+R)/2):
                R=((L+R)/2)
        else:
            # if L != (((L+R)/2)):
                L=((L+R)/2)
    BNalg.run(g,root,k,(vec[int((L))]),Avilabilty)
    if g.nodes[0]['minSend']['k'+str(k)]['bn']:
        return vec[int(L)]
    return vec[int(R)]
        
    
    # LoadSum=sumLoad(g)
    # minLink,maxLink = minMaxLink(g)
    # print("minlink: "+str(minLink)+" maxLink: "+str(maxLink)+" load: "+str(LoadSum))
    # # BNalg.run(g,root,k,int(LoadSum/2))
    # L=1/maxLink
    # R=LoadSum/minLink
    # # rounds=0
    # while L <= R and  (R-L)>= 1/maxLink:#rounds<2:
    #     Avilabilty=AvalbiltyCalc(g,cap)
    #     BNalg.run(g,root,k,((L+R)/2),Avilabilty)
    #     print('X: '+str((L+R)/2)+' R: '+str(R)+' L: '+str(L))
    #     if g.nodes[0]['minSend']['k'+str(k)]['bn']:
    #         # if R != ((L+R)/2):
    #             R=((L+R)/2)
    #     else:
    #         # if L != (((L+R)/2)):
    #             L=(((L+R)/2))
    # # for i in range(0,LoadSum+1):
    # #     BNalg.run(g,root,k,i)
    # #     if 
    # BNalg.run(g,root,k,L,Avilabilty)
    # if g.nodes[0]['minSend']['k'+str(k)]['bn']:
    #     return L
    # return R

def SerchVector(g,root):
    LoadSum=sumLoad(g)
    wieghts=[]
    x=[]
    for e in g.edges:
        if not ( g.edges[e]['Wieght'] in wieghts):
            wieghts.append( g.edges[e]['Wieght'])
            for i in range(1,LoadSum+1):
                tmp=i/g.edges[e]['Wieght']
                if not tmp in x:
                    x.append(tmp)
    x=sorted(x)
    return x
            

def Add_InNetwork_Capacity(g):
   for node in g.nodes:
       att={node:{'Jobs':{'number':0,'list':[]}}}
       att[node]['color']='red'
       nx.set_node_attributes(g,att)

def AvalbiltyCalc(g,cap):
    avalbilty=[]
    for node in g.nodes:
        if g.nodes[node]['Jobs']['number'] < cap:
            avalbilty.append(True)
        else:
            avalbilty.append(False)
    return avalbilty
        
def BottleNeck_Vs_load(deg,h,k):
    deg=2
    h=7
    k=8
    sample=[1,2,4,8,16,32,64]
    distrbution=paseDistrebution('distrebutions.txt')


    for k in sample:
        BootleNALG={}
        LoadALG={}
        for dis in distrbution:
            BootleNALG[dis[0]]={'SumMessage': 0, 'BottleNeck': 0}
            LoadALG[dis[0]]={'SumMessage': 0, 'BottleNeck': 0}
        
            g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
            loadDist=list(np.random.permutation(dis[1]))
            root=len(g.nodes)
            g.add_edge(len(g.nodes),0)
        
        
            leafL=leafList(g)
            addLoad(g,loadDist,leafL)
        
        
            X=findX(g,root,k)
            BNalg.run(g,root,k,X)
        # GtoFile(g,"Eample_X_"+str(X))
            BNalg.NewColoring(g,root,root,k,X)
            BNalg.messageCount(g,root)
            SumMessage,BottleNeck=maxBottleNeck_messageNum(g)
            BootleNALG[dis[0]]['SumMessage']=SumMessage
            BootleNALG[dis[0]]['BottleNeck']=BottleNeck
        # plt.figure(0)
        # plt.title('Toy example,BN alg, number of messages: '+str(SumMessage)+' ,bottle neck: '+str(BottleNeck))
        # labels = nx.get_node_attributes(g, 'load') 
        # # nx.draw(g,pos=graphviz_layout(g, prog="dot"),with_labels=True)
        # # nx.draw(g,pos=graphviz_layout(g, prog="dot"),labels=labels)
        # nx.draw(g,pos=graphviz_layout(g, prog='dot'),node_color=colorMap(g),labels=labels)
        # edge_labels = nx.get_edge_attributes(g,'mesageCount')
        # pos=graphviz_layout(g,prog='dot')
        # nx.draw_networkx_edge_labels(g, pos, edge_labels = edge_labels,rotate=False)
        
        
        
            Loadg=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
        # loadDist=[1,2,1,2]
            root=len(Loadg.nodes)
            Loadg.add_edge(len(Loadg.nodes),0)
        
            leafL=leafList(Loadg)
            addLoad(Loadg,loadDist,leafL)
        
        # plt.savefig("exaple.png")
            Lalg.run(Loadg,root,k)
        # GtoFile(g,"Eample_X_"+str(X))
            Lalg.NewColoring(Loadg,root,root,0,k)
            BNalg.messageCount(Loadg,root)
            SumMessage,BottleNeck=maxBottleNeck_messageNum(Loadg)
            LoadALG[dis[0]]['SumMessage']=SumMessage
            LoadALG[dis[0]]['BottleNeck']=BottleNeck
        
        # plt.figure(1)
        # plt.title('Toy example,Load alg, number of messages: '+str(SumMessage)+' ,bottle neck: '+str(BottleNeck))
        # labels = nx.get_node_attributes(Loadg, 'load') 
        # # nx.draw(g,pos=graphviz_layout(g, prog="dot"),with_labels=True)
        # # nx.draw(g,pos=graphviz_layout(g, prog="dot"),labels=labels)
        # nx.draw(g,pos=graphviz_layout(Loadg, prog='dot'),node_color=colorMap(Loadg),labels=labels)
        # edge_labels = nx.get_edge_attributes(Loadg,'mesageCount')
        # pos=graphviz_layout(Loadg,prog='dot')
        # nx.draw_networkx_edge_labels(Loadg, pos, edge_labels = edge_labels,rotate=False)
        
        distList=list(LoadALG.keys())
        x=np.arange(len(distList))
        width = 0.35
        fig, ax = plt.subplots(1,2)
        rects1 = ax[0].bar(x - width/2, [BootleNALG[d]['SumMessage'] for d in distList], width, label='BottelNeck Alg')
        rects2 = ax[0].bar(x + width/2, [LoadALG[d]['SumMessage'] for d in distList], width, label='Load Alg')
        ax[0].set_ylabel('# messeges')
        ax[0].set_title('Bottleneck Alg VS Load Alg message count k: '+str(k))
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(distList)
        ax[0].legend()
        
        # ax[0].bar_label(rects1, padding=3)
        # ax[0].bar_label(rects2, padding=3)
        
        fig.tight_layout()
        
        
        # fig, ax = plt.subplots()
        rects1 = ax[1].bar(x - width/2, [BootleNALG[d]['BottleNeck'] for d in distList], width, label='BottelNeck Alg')
        rects2 = ax[1].bar(x + width/2, [LoadALG[d]['BottleNeck'] for d in distList], width, label='Load Alg')
        ax[1].set_ylabel('Max bottleneck')
        ax[1].set_title('Bottleneck Alg VS Load Alg max bottleneck k: '+str(k))
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(distList)
        ax[1].legend()
        
        # ax[1].bar_label(rects1, padding=3)
        # ax[1].bar_label(rects2, padding=3)
        
        fig.tight_layout()

def wieghtFunction(func,i):
    if func == 'linear':
        return 1+i
    if func == 'power':
        return 1.5**i
    if func == 'uniform':
        return 1

def AddWieghtToEges(g,root,func):
    l=list (nx.all_pairs_dijkstra_path_length(g))
    rootIndex=list(g.nodes).index(root)
    depth=max(l[rootIndex][1].values())
    r={}
    for i in range(0,depth+1):
        r[i]=[]
    for node in g.nodes():
        r[l[rootIndex][1][node]].append(node)
    
    # for node in r[3]:    
    #     nodeRun(g,node,0,2)
    #     print(node)
    # for node in r[2]:    
    #     nodeRun(g,node,0,2)
    #     print(node)
        
    for i in range(0,depth+1):
        # print('i: '+str(i))
        for node in r[depth-i]: 
            # if g.nodes[node]['type'] == 's':
            #     # print(node)
            if node == root:
                return
            perent=list(g.in_edges(nbunch=node))[0][0]
            att={(perent,node):{'Wieght': wieghtFunction(func, i)}}
            nx.set_edge_attributes(g,att)
def CalcMeanVar(messageExp,scale):
    x=[]
    for i in range(0,len(messageExp[0])):
        y=[]
        for j in range(0,len(messageExp)):
            y.append(messageExp[j][i]/scale[j])
        x.append(y)
    #compute mean and var
    mean=[]
    var=[]
    for i in x:
        mean.append(np.mean(i) )
        var.append(np.sqrt(np.var(i)))
    return [mean,var]

def CalcMeanVarCap(messageExp,scale):
    x=[]
    for i in range(0,len(messageExp[0])):
        y=[]
        for j in range(0,len(messageExp)):
            y.append(messageExp[j][i]/scale[j][i])
        x.append(y)
    #compute mean and var
    mean=[]
    var=[]
    for i in x:
        mean.append(np.mean(i) )
        var.append(np.sqrt(np.var(i)))
    return [mean,var]

def plot_coloring(gr,Blist):
    gt=gr.copy()
    for node in gr.nodes:
        gt.nodes[node]['color']='red'
    for node in Blist:
        gt.nodes[node]['color']='blue'
    #gr.nodes[0]['color']='green'
    return gt
def writeToFile(name,List):
    with open(name+'.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % place for place in List)
def JobColor(g,root,k,load,avalabilty):
    leafL=leafList(g)
    addLoad(g,load,leafL)
    X=findX(g,root,k,avalabilty)
                
    BNalg.run(g,root,k,X,avalabilty)
    coloring=BNalg.NewColoring(g,root,root,k,X)
    return [g,coloring]
def TopJobColor(g,root,k,load,Avalibily):
    leafL=leafList(g)
    addLoad(g,load,leafL)
    top=[]
    for node in g.nodes:
        if Avalibily[node]:
            k=k-1
            top.append(node)
        if k==0:
            break
    g=plot_coloring(g,top)
    return [g,top]
        

def MaxColorJob(amount,g,leafList,Avalibily):
    deg=[]
    for leaf in leafList:
        deg.append(g.nodes[leaf]['load'])
    l=[]
    for i in leafList:
        if len(l)<amount:
            if Avalibily[leafList[deg.index(max(deg))]]:
                 l.append(leafList[deg.index(max(deg))])
            deg[deg.index(max(deg))]=0
    return l

def MaxJobColor(g,root,k,load,Avalibily):
    leafL=leafList(g)
    addLoad(g,load,leafL)
    MaxC=MaxColorJob(k,g,leafL,Avalibily)
    g=plot_coloring(g,MaxC)
    return [g,MaxC]
    
def plot_coloringJob(gr,Blist,job):
    # for node in gr.nodes:
    #     gr.nodes[node]['color']='red'
    for node in Blist:
        gr.nodes[node]['color']='blue'
        gr.nodes[node]['Jobs']['number']=gr.nodes[node]['Jobs']['number']+1
        gr.nodes[node]['Jobs']['list'].append(job)
        
    #gr.nodes[0]['color']='green'
    return gr
def pickRamdonLevel(level,Avalibily,k):
    x=[]
    levelC=LevelColor(level)
    while True:
        for i in levelC:
            if Avalibily[i]:
                x.append(i)
        if len(x)>=k:
            return random.sample(x, k)
        else:
            level=level+1
            levelC=LevelColor(level)
        
    

def LevelJobColor(g,root,k,load,Avalibily):
    leafL=leafList(g)
    addLoad(g,load,leafL)
    level=int(math.log(k,2))
    levelC=pickRamdonLevel(level,Avalibily,k)
    # levelC=LevelColor(level)
    # flag=True
    # while flag:
    #     count=0
    #     for node in levelC:
    #         if Avalibily[node] == False:
    #             level=level+1
    #             levelC=LevelColor(level)
    #             count=0
    #             break
    #         count=count+1
    #     if count == len(levelC):
    #         flag=False
    g=plot_coloring(g,levelC)
    return [g,levelC]

def BottleNeckArray(Jobsgraph):
    bottleneck=[]
    edges={}
    for e in Jobsgraph[0].edges():
        edges[e]=0
    for gS in Jobsgraph:
        Walg.messageCount(gS,0)
        NetworkUtiliztion(gS)
        for e in gS.edges:
            edges[e]=edges[e]+gS.edges[e]['mesageCount']/gS.edges[e]['Wieght']
        bottleneck.append(max(edges.values()))
    return bottleneck

def AlgVS(weight,expNum):
    deg=2
    h=7
    k=256
    messageExpAlg=[]
    messageExpTop=[]
    messageExpLevel=[]
    messageExpMax=[]
    BnExpAlg=[]
    BnExpTop=[]
    BnExpLevel=[]
    BnExpMax=[]
    messgageNumZero=[]
    BnCountZero=[]
    messgageNumOpt=[]
    BnCountOpt=[]
    
    sample=[1,2,4,8,16,32]
    distrbution=paseDistrebution('distrebutions.txt')
    cap=1
    # name='wietghed_linear_distrebutionRuns_5_tree_256'
    name='Bottleneck_wietghed_'+weight+'_distrebutionRuns_'+str(expNum)+'_tree_256'
    if not(os.path.isdir(name)):
        os.mkdir(name)
    os.chdir('.//'+name)
    D=distrbution
    # func="uniform"
    for dis in D:
        if not(os.path.isdir(dis[0])):
            os.mkdir(dis[0])    
        os.chdir('.//'+dis[0])
        deg=2
        h=7
        k=max(sample)
        messageExpAlg=[]
        messageExpTop=[]
        messageExpLevel=[]
        messageExpMax=[]
        BnExpAlg=[]
        BnExpTop=[]
        BnExpLevel=[]
        BnExpMax=[]
        messgageNumZero=[]
        BnCountZero=[]
        messgageNumOpt=[]
        BnCountOpt=[]
        
        for j in range (0,expNum):
            loadDist=list(np.random.permutation(dis[1]))
            g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
            root=0
            leafL=leafList(g)
            addLoad(g,loadDist,leafL)
            AddWieghtToEges(g,root,weight)
            Add_InNetwork_Capacity(g)
            Avilabilty=AvalbiltyCalc(g,1)
            # X=findX(g,root,k,1)
                
            # BNalg.run(g,root,k,X,Avilabilty)
            # Walg.run(g,root,k)
            nx.write_gpickle(g,"binaryTree-"+str(len(g.nodes))+"_nodes_"+"_distebution_"+dis[0]+"_run_"+str(j)+".gpickle")
            
            #Alg sample
            messgageList=[]
            BnList=[]
            for i in sample:#[1,2,3,4,7,8,15,16,31,32]:
                gr=copy.deepcopy(g)
                Avilabilty=AvalbiltyCalc(g,cap)
                X=findX(gr,root,i,Avilabilty)
                
                BNalg.run(gr,root,i,X,Avilabilty)
                BNalg.NewColoring(gr,root,root,i,X)
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                if not(X==BottleNeck):
                    print('------Error-------')
                    print('k: '+str(i)+' X: '+str(X)+' BottleNeck: '+str(BottleNeck))
                    
                messgageList.append(SumMessage)
                BnList.append(BottleNeck)
            messageExpAlg.append(messgageList)
            BnExpAlg.append(BnList)
    
            #level sample
            messgageList=[]
            BnList=[]
            for i in sample:
                gr=copy.deepcopy(g)
                gr=plot_coloring(gr,LevelColor(int(math.log(i,2))))
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageList.append(SumMessage)
                BnList.append(BottleNeck)
            messageExpLevel.append(messgageList)
            BnExpLevel.append(BnList)
            
            #level Top
            messgageList=[]
            BnList=[]
            for i in sample:#[1,3,7,15,31]:
                gr=copy.deepcopy(g)
                gr=plot_coloring(gr,TopColor(i,gr, leafL))
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageList.append(SumMessage)
                BnList.append(BottleNeck)
            messageExpTop.append(messgageList)
            BnExpTop.append(BnList)        
            #level Max
            messgageList=[]
            BnList=[]
            for i in sample:
                gr=copy.deepcopy(g)
                gr=plot_coloring(gr,MaxColor(i, gr, leafL))
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageList.append(SumMessage)
                BnList.append(BottleNeck)
            messageExpMax.append(messgageList)
            BnExpMax.append(BnList)
            
            #All red
            gr=copy.deepcopy(g)
            gr=plot_coloring(gr,[])
            Walg.messageCount(gr,root)
            SumMessage,BottleNeck=NetworkUtiliztion(gr)
            messgageNumZero.append(SumMessage)
            BnCountZero.append(BottleNeck)
            #add consitncy check
            
            #All blue
            gr=copy.deepcopy(g)
            gr=plot_coloring(gr,[ x for x in g.nodes()])
            Walg.messageCount(gr,root)
            SumMessage,BottleNeck=NetworkUtiliztion(gr)
            messgageNumOpt.append(SumMessage)
            BnCountOpt.append(BottleNeck)
        writeToFile("messageExpAlg",messageExpAlg)
        writeToFile("BnExpAlg",BnExpAlg)
        writeToFile("messageExpTop",messageExpTop)
        writeToFile("BnExpTop",BnExpTop)
        writeToFile("messageExpMax",messageExpMax)
        writeToFile("BnExpMax",BnExpMax)
        writeToFile("messageExpLevel",messageExpLevel)
        writeToFile("BnExpLevel",BnExpLevel)
        writeToFile("BnCountZero",BnCountZero)
        writeToFile("BnCountOpt",BnCountOpt)
        writeToFile("messgageNumOpt",messgageNumOpt)
        writeToFile("messgageNumZero",messgageNumZero)
        os.chdir('..')
    os.chdir('..')
        

def AlgVS_plots(dstDir,path,distrebutions,wieght,scale='allRed'):
    if not(os.path.isdir(dstDir)):
        os.mkdir(dstDir)
    os.chdir('.//'+dstDir)
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_linear_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\distrebutionRuns_paremeterServer_range_10000_lenght_5000_new3\\"
    # file=open(path+'\\Results.txt')
    # res=ast.literal_eval(file.read())
    # distrebutions=['Mixed','Uniform1','Uniform2','Skewed','PowerLaw1','PowerLaw3','PowerLaw2']
    distrebutions=['Uniform2','PowerLaw1']
    for disrebution in distrebutions:
        directory_contents=os.listdir(path+'\\'+disrebution)
        data={}
        for f in directory_contents:
            try:
                if f.split('.')[1] == 'txt':
                    file=open(path+'\\'+disrebution+'\\'+f)
                    key=f.split('.')[0]
                    data[key]=[]
                    lines=file.readlines()
                    for line in lines:
                        data[key].append(ast.literal_eval(line))
                        
            except:
                print()
                
        sample=[1,2,4,8,16,32]
        s=[]    
        for i in data['BnCountZero']:
            st=[]
            for j in sample:
                st.append(i)
            s.append(st)
        scaleFactor=s
        M=np.power(10,6)
        
        scaleFactor=1#sum(data['messgageNumZero'])/len(data['messgageNumZero'])
        if scale == 'alg':
            scaleFactor=data['BnExpAlg']
        else:
            scaleFactor=s

           
        sF=scaleFactor
        plt.close(1)
        plt.figure(1)
        ax = plt.gca()
        ax.set_box_aspect(1/2)
        plt.tight_layout()
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
        # plt.plot([0,1,3,7,15,32],[(sum(data['BnCountOpt'])/len(data['BnCountOpt']))/(sum(data['BnCountZero'])/len(data['BnCountZero'])) for x in range(0,len(sample))],'b',lw=3,label='All blue',)
        # plt.plot([0,1,3,7,15,32],[1 for x in range(0,len(sample))],'r',lw=3,label='All red')

        # if scale == 'alg':
        #     s=[]    
        #     for x in scaleFactor:
        #         s.append([x[i] for i in [0,1,3,5,7,9]])
        #     sF=s
        # else:
        #     sF=scaleFactor
        meanLevel,varLevel=CalcMeanVarCap(data['BnExpLevel'],sF)
        plt.errorbar(sample, [x for x in meanLevel], yerr=[x for x in varLevel],lw=3,ls='-.',markersize=12,marker='x',label='Level')
        
        meanMax,varMax=CalcMeanVarCap(data['BnExpMax'],sF)
        plt.errorbar(sample, [x for x in meanMax], yerr=[x for x in varMax],ls=':',lw=3,marker='s',markersize=12,label='Max')
        # if scale == 'alg':
        #     s=[]    
        #     for x in scaleFactor:
        #         s.append([x[i] for i in [0,1,3,5,7,9]])
        #     sF=s
        # else:
        #     sF=scaleFactor
        meanTop,varTop=CalcMeanVarCap(data['BnExpTop'],sF)
        plt.errorbar(sample, [x for x in meanTop], yerr=[x for x in varTop],lw=3,ls='--',marker='^',markersize=12,label='Top')


        # meanAlg,varAlg=CalcMeanVarCap(data['BnCountZero'],scaleFactor)
        # plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls='--',color='r',markersize=12,marker='-',label='All red')
        # plt.plot(sample,BnExpLevel[0],'-',marker='x',label='Level')
        # plt.plot(sample,BnExpMax[0],'-.',marker='s',label='Max')
        # plt.plot([1,3,7,15,31],BnExpTop[0],'--',marker='^',label='Top')
        plt.grid(True, which="both", ls="-")
        plt.xlabel(r"Number of blue nodes ($k$)",fontsize=18)
        plt.ylabel(r'Normalized Congestion',fontsize=18)
        if scale == 'alg':
            Loc='upper left'
            meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],color='tab:purple',lw=3,ls='-',markersize=12,label='SMC')
            meanAlg,varAlg=CalcMeanVarCap(s,scaleFactor)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,color='r',ls=':',markersize=12,label='All red')
            ax.set_yscale('log',base=2)
            ax.set_ylim(0,16)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        else:
            Loc='upper right'
            plt.plot(sample,[1 for x in range(0,len(sample))],'r',lw=3,label='All red')
            ax.set_ylim(ymin=0)
            
            meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],sF)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls='--',markersize=12,marker='o',label='SMC')
            # ax.set_yscale('log',base=2)
            ax.set_ylim(0)
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        ax.set_xlim(xmin=0)
       
        
        
        plt.legend(loc=Loc,fontsize=12,ncol=2)
        # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        # plt.savefig("Weighted_Util_Multi_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
        plt.savefig("Weighted_Util_"+wieght+"_count_distribution_"+disrebution+"_scale_"+scale+"_SOAR_bold.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')


def MultiJobsMultiCap(weight,expNum,caps):
    deg=2
    h=7
    k=256
    messageExpAlg=[]
    messageExpTop=[]
    messageExpLevel=[]
    messageExpMax=[]
    BnExpAlg=[]
    BnExpTop=[]
    BnExpLevel=[]
    BnExpMax=[]
    messgageNumZero=[]
    BnCountZero=[]
    messgageNumOpt=[]
    BnCountOpt=[]
    
    sample=[1,2,4,8,16,32]
    distrbution=paseDistrebution('distrebutions.txt')
    
    name='Multi_MultiCapT'+str(max(caps))+'_load_'+str(weight)+'_distrebutionRuns_'+str(expNum)+'_tree_256'
    
    if not(os.path.isdir(name)):
        os.mkdir(name)
    os.chdir('.//'+name)
    D=[['Coin',[]]]
    # D=[distrbution[0],distrbution[1],distrbution[2],distrbution[6]]
    for dis in D:
        if not(os.path.isdir(dis[0])):
            os.mkdir(dis[0])    
        os.chdir('.//'+dis[0])
        deg=2
        h=7
        k=16
        cap=4
        root=0
        
        messageExpAlgT=[]
        messageExpTopT=[]
        messageExpMaxT=[]
        messageExpLevelT=[]
        messgageNumOptT=[]
        messgageNumZeroT=[]
        BnExpAlgT=[]
        BnExpTopT=[]
        BnExpLevelT=[]
        BnExpMaxT=[]
        BnCountZeroT=[]
        BnCountOptT=[]
        for e in range(0,expNum):
            messageExpAlg=[]
            messageExpTop=[]
            messageExpLevel=[]
            messageExpMax=[]
            BnExpAlg=[]
            BnExpTop=[]
            BnExpLevel=[]
            BnExpMax=[]
        
            
            messgageNumZero=[]
            BnCountZero=[]
            messgageNumOpt=[]
            BnCountOpt=[]
            for cap in caps:
                gSOAR=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
                Add_InNetwork_Capacity(gSOAR)
                AddWieghtToEges(gSOAR,root,weight)
                gMax=copy.deepcopy(gSOAR)
                gLevel=copy.deepcopy(gSOAR)
                gTop=copy.deepcopy(gSOAR)
                AvalibilySOAR=AvalbiltyCalc(gSOAR,cap)
                AvalibilyMax=AvalbiltyCalc(gMax,cap)
                AvalibilyLevel=AvalbiltyCalc(gLevel,cap)
                AvalibilyTop=AvalbiltyCalc(gTop,cap)
                JobsgraphSOAR=[]
                JobsUtilSOAR=[]
                JobsUtilLevel=[]
                JobsgraphLevel=[]
                JobsgraphTop=[]
                JobsUtilTop=[]
                JobsgraphMax=[]
                JobsUtilMax=[]
                JobsUtilZero=[]
                for j in range (0,32):
                    # loadDist=list(np.random.permutation(dis[1]))
                    if (random.randint(0, 1)==1):
                        loadDist=list(np.random.permutation(distrbution[1][1]))
                    else:
                        loadDist=list(np.random.permutation(distrbution[2][1]))
                    g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
                    AddWieghtToEges(g,root,weight)
                    Add_InNetwork_Capacity(g)
                    
                    #SOAR
                    gr=copy.deepcopy(g)
                    AvalibilySOAR=AvalbiltyCalc(gSOAR,cap)
                    gr,c=JobColor(gr,root,k,loadDist,AvalibilySOAR)
                    JobsgraphSOAR.append(gr)
                    # JobsUtilSOAR.append(gr.nodes[root]['minSend']['l0']['k'+str(k)])
                    gSOAR=plot_coloringJob(gSOAR,c,len(JobsgraphSOAR))
                
                
                    
                    #level
                    gr=copy.deepcopy(g)
                    AvalibilyLevel=AvalbiltyCalc(gLevel,cap)
                    gr,c=LevelJobColor(gr,root,k,loadDist,AvalibilyLevel)
                    JobsgraphLevel.append(gr)
                    Walg.messageCount(gr,root)
                    SumMessage,BottleNeck=NetworkUtiliztion(gr)
                    gLevel=plot_coloringJob(gLevel,c,len(JobsgraphLevel))
                    JobsUtilLevel.append(SumMessage)
                    
                
            
                    
                    #Top
                    gr=copy.deepcopy(g)
                    AvalibilyTop=AvalbiltyCalc(gTop,cap)
                    gr,c=TopJobColor(gr,root,k,loadDist,AvalibilyTop)
                    JobsgraphTop.append(gr)
                    Walg.messageCount(gr,root)
                    SumMessage,BottleNeck=NetworkUtiliztion(gr)
                    gTop=plot_coloringJob(gTop,c,len(JobsgraphTop))
                    JobsUtilTop.append(SumMessage)
                    
                
                    
                    #Max
                    gr=copy.deepcopy(g)
                    AvalibilyMax=AvalbiltyCalc(gMax,cap)
                    gr,c=MaxJobColor(gr,root,k,loadDist,AvalibilyMax)
                    JobsgraphMax.append(gr)
                    Walg.messageCount(gr,root)
                    SumMessage,BottleNeck=NetworkUtiliztion(gr)
                    gMax=plot_coloringJob(gMax,c,len(JobsgraphMax))
                    JobsUtilMax.append(SumMessage)
                                    
                    gr=copy.deepcopy(g)
                    leafL=leafList(gr)
                    addLoad(gr,loadDist,leafL)
                    gr=plot_coloring(gr,[])
                    JobsUtilZero.append(gr)
                    Walg.messageCount(gr,root)
                    SumMessage,BottleNeck=NetworkUtiliztion(gr)
                    messgageNumZero.append(SumMessage)
                    # BnCountZero.append(BottleNeck)
                    
                    
                messageExpLevel.append(sum(JobsUtilLevel))
                messageExpAlg.append(sum(JobsUtilSOAR))
                messageExpMax.append(sum(JobsUtilMax))
                messageExpTop.append(sum(JobsUtilTop))
                x=BottleNeckArray(JobsgraphSOAR)
                BnExpAlg.append(x[len(x)-1])
                x=BottleNeckArray(JobsgraphTop)
                BnExpTop.append(x[len(x)-1])
                x=BottleNeckArray(JobsgraphLevel)
                BnExpLevel.append(x[len(x)-1])
                x=BottleNeckArray(JobsgraphMax)
                BnExpMax.append(x[len(x)-1])
                x=BottleNeckArray(JobsUtilZero)
                BnCountZero.append(x[len(x)-1])
                    
                    #All red

                    #add consitncy check
                    
                    #All blue
                gr=copy.deepcopy(g)
                leafL=leafList(gr)
                addLoad(gr,loadDist,leafL)
                gr=plot_coloring(gr,[ x for x in g.nodes()])
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageNumOpt.append(SumMessage)
                BnCountOpt.append(BottleNeck)
            messageExpAlgT.append(messageExpAlg)
            messageExpTopT.append(messageExpTop)
            messageExpMaxT.append(messageExpMax)
            messageExpLevelT.append(messageExpLevel)
            messgageNumOptT.append(messgageNumOpt)
            messgageNumZeroT.append(messgageNumZero)
            BnExpAlgT.append(BnExpAlg)
            BnExpTopT.append(BnExpTop)
            BnExpLevelT.append(BnExpLevel)
            BnExpMaxT.append(BnExpMax)
            BnCountOptT.append(BnCountOpt)
            BnCountZeroT.append(BnCountZero)
        writeToFile("messageExpAlg",messageExpAlgT)
        writeToFile("BnExpAlg",BnExpAlgT)
        writeToFile("messageExpTop",messageExpTopT)
        writeToFile("BnExpTop",BnExpTopT)
        writeToFile("messageExpMax",messageExpMaxT)
        writeToFile("BnExpMax",BnExpMaxT)
        writeToFile("messageExpLevel",messageExpLevelT)
        writeToFile("BnExpLevel",BnExpLevelT)
        writeToFile("BnCountZero",BnCountZeroT)
        writeToFile("BnCountOpt",BnCountOpt)
        writeToFile("messgageNumOpt",messgageNumOptT)
        writeToFile("messgageNumZero",messgageNumZeroT)
        os.chdir('..')
    os.chdir('..')


def MutliJobFixCapAlgVS(weight,expNum,cap):
    deg=2
    h=7
    k=256
    messageExpAlg=[]
    messageExpTop=[]
    messageExpLevel=[]
    messageExpMax=[]
    BnExpAlg=[]
    BnExpTop=[]
    BnExpLevel=[]
    BnExpMax=[]
    messgageNumZero=[]
    BnCountZero=[]
    messgageNumOpt=[]
    BnCountOpt=[]
    
    sample=[1,2,4,8,16,32]
    distrbution=paseDistrebution('distrebutions.txt')
    
    name='Multi_FixedCapT'+str(cap)+'_load_'+str(weight)+'_distrebutionRuns_'+str(expNum)+'_tree_256'
    
    if not(os.path.isdir(name)):
        os.mkdir(name)
    os.chdir('.//'+name)
    D=[distrbution[0],distrbution[1],distrbution[2],distrbution[6]]
    D=[1]
    
    for dis in D:
        # if not(os.path.isdir(dis[0])):
        #     os.mkdir(dis[0])
        if not(os.path.isdir('Coin')):
            os.mkdir('Coin')    
        # os.chdir('.//'+dis[0])
        os.chdir('.//'+'Coin')
        deg=2
        h=7
        k=16
        root=0
        messageExpAlgT=[]
        messageExpTopT=[]
        messageExpMaxT=[]
        messageExpLevelT=[]
        messgageNumOptT=[]
        messgageNumZeroT=[]
        BnExpAlgT=[]
        BnExpTopT=[]
        BnExpLevelT=[]
        BnExpMaxT=[]
        BnCountZeroT=[]
        BnCountOptT=[]
        for e in range(0,expNum):
            messageExpAlg=[]
            messageExpTop=[]
            messageExpLevel=[]
            messageExpMax=[]
            BnExpAlg=[]
            BnExpTop=[]
            BnExpLevel=[]
            BnExpMax=[]
            gSOAR=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
            Add_InNetwork_Capacity(gSOAR)
            AddWieghtToEges(gSOAR,root,weight)
            gMax=copy.deepcopy(gSOAR)
            gLevel=copy.deepcopy(gSOAR)
            gTop=copy.deepcopy(gSOAR)
            AvalibilySOAR=AvalbiltyCalc(gSOAR,cap)
            AvalibilyMax=AvalbiltyCalc(gMax,cap)
            AvalibilyLevel=AvalbiltyCalc(gLevel,cap)
            AvalibilyTop=AvalbiltyCalc(gTop,cap)
            JobsgraphSOAR=[]
            JobsUtilSOAR=[]
            JobsUtilLevel=[]
            JobsgraphLevel=[]
            JobsgraphTop=[]
            JobsUtilTop=[]
            JobsgraphMax=[]
            JobsUtilMax=[]
            JobsUtilZero=[]
            
            messgageNumZero=[]
            BnCountZero=[]
            messgageNumOpt=[]
            BnCountOpt=[]
        
            for j in range (0,32):
                print('Jobs: '+str(j))
                # loadDist=list(np.random.permutation(dis[1]))
                if (random.randint(0, 1)==1):
                    loadDist=list(np.random.permutation(distrbution[1][1]))
                else:
                    loadDist=list(np.random.permutation(distrbution[2][1]))
                g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
                AddWieghtToEges(g,root,weight)
                Add_InNetwork_Capacity(g)
                
                #SOAR
                gt=copy.deepcopy(g)
                AvalibilySOAR=AvalbiltyCalc(gSOAR,cap)
                gr,c=JobColor(gt,root,k,loadDist,AvalibilySOAR)
                JobsgraphSOAR.append(gr)
                JobsUtilSOAR.append(gr.nodes[root]['minSend']['k'+str(k)]['U'])
                gSOAR=plot_coloringJob(gSOAR,c,len(JobsgraphSOAR))
                # SumMessage,BottleNeck=NetworkUtiliztion(gSOAR)
            
                messageExpAlg.append(sum(JobsUtilSOAR))
                
                
                #level
                print('level')
                gr=copy.deepcopy(g)
                AvalibilyLevel=AvalbiltyCalc(gLevel,cap)
                gr,c=LevelJobColor(gr,root,k,loadDist,AvalibilyLevel)
                JobsgraphLevel.append(gr)
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                gLevel=plot_coloringJob(gLevel,c,len(JobsgraphLevel))

                
                messageExpLevel.append(sum(JobsUtilLevel))
        
                
                #Top
                print('top')
                gr=copy.deepcopy(g)
                AvalibilyTop=AvalbiltyCalc(gTop,cap)
                gr,c=TopJobColor(gr,root,k,loadDist,AvalibilyTop)
                JobsgraphTop.append(gr)
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                gTop=plot_coloringJob(gTop,c,len(JobsgraphTop))
                JobsUtilTop.append(SumMessage)

                
                messageExpTop.append(sum(JobsUtilTop))
                
                #Max
                print('max')
                # if j==31:
                #     print('here')
                gr=copy.deepcopy(g)
                AvalibilyMax=AvalbiltyCalc(gMax,cap)
                gr,c=MaxJobColor(gr,root,k,loadDist,AvalibilyMax)
                JobsgraphMax.append(gr)
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                gMax=plot_coloringJob(gMax,c,len(JobsgraphMax))
                JobsUtilMax.append(SumMessage)

                messageExpMax.append(sum(JobsUtilMax))
                
                #All red
                gr=copy.deepcopy(g)
                leafL=leafList(gr)
                addLoad(gr,loadDist,leafL)
                gr=plot_coloring(gr,[])
                JobsUtilZero.append(gr)
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageNumZero.append(SumMessage)
                BnCountZero.append(BottleNeck)
                #add consitncy check
                
                #All blue
                gr=copy.deepcopy(g)
                leafL=leafList(gr)
                addLoad(gr,loadDist,leafL)
                gr=plot_coloring(gr,[ x for x in g.nodes()])
                Walg.messageCount(gr,root)
                SumMessage,BottleNeck=NetworkUtiliztion(gr)
                messgageNumOpt.append(SumMessage)
                BnCountOpt.append(BottleNeck)
                
            messageExpAlgT.append(messageExpAlg)
            messageExpTopT.append(messageExpTop)
            messageExpMaxT.append(messageExpMax)
            messageExpLevelT.append(messageExpLevel)
            messgageNumOptT.append(messgageNumOpt)
            messgageNumZeroT.append(messgageNumZero)
            BnExpAlgT.append(BottleNeckArray(JobsgraphSOAR))
            BnExpTopT.append(BottleNeckArray(JobsgraphTop))
            BnExpLevelT.append(BottleNeckArray(JobsgraphLevel))
            BnExpMaxT.append(BottleNeckArray(JobsgraphMax))
            BnCountOptT.append(BnCountOpt)
            BnCountZeroT.append(BottleNeckArray(JobsUtilZero))
        writeToFile("messageExpAlg",messageExpAlgT)
        writeToFile("BnExpAlg",BnExpAlgT)
        writeToFile("messageExpTop",messageExpTopT)
        writeToFile("BnExpTop",BnExpTopT)
        writeToFile("messageExpMax",messageExpMaxT)
        writeToFile("BnExpMax",BnExpMaxT)
        writeToFile("messageExpLevel",messageExpLevelT)
        writeToFile("BnExpLevel",BnExpLevelT)
        writeToFile("BnCountZero",BnCountZeroT)
        writeToFile("BnCountOpt",BnCountOptT)
        writeToFile("messgageNumOpt",messgageNumOptT)
        writeToFile("messgageNumZero",messgageNumZeroT)
        os.chdir('..')
    os.chdir('..')
    # os.chdir('..')

def MutliJobMultiCapAlgVS_plots(dstDir,path,distrebutions,wieght):
    os.chdir('.\\'+dstDir)
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap_load_power_distrebutionRuns_10_tree_256\\"
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap_load_uni_Coin_distrebutionRuns_10_tree_256"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_linear_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_Power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\Multi_load_wieght_1_distrebutionRuns_tree_256\\"
    
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap32_load_uni_distrebutionRuns_10_tree_256\\"
    # file=open(path+'\\Results.txt')
    # res=ast.literal_eval(file.read())
    # distrebutions=['Mixed','Uniform1','Uniform2','PowerLaw1']
    # distrebutions=['Coin']
    for disrebution in distrebutions:
        directory_contents=os.listdir(path+'\\'+disrebution)
        data={}
        for f in directory_contents:
            try:
                if f.split('.')[1] == 'txt':
                    file=open(path+'\\'+disrebution+'\\'+f)
                    key=f.split('.')[0]
                    data[key]=[]
                    lines=file.readlines()
                    for line in lines:
                        data[key].append(ast.literal_eval(line))
                        
            except:
                print()
        
        M=np.power(10,6)
        # scaleFactor=[data['BnCountZero'][0][i] for i in range(1,len(data['BnCountZero'][0])+1)]
        # sample=[4,8,16]#[i for i in range(1,len(data['messageExpMax'])+1)]
        sample=[4,8,16,32]
        # n=len(data['messgageNumZero'])
        scaleFactor=data['BnCountZero']
        plt.close(1)
        plt.figure(1)
        ax = plt.gca()
        ax.set_box_aspect(1/2)
        plt.tight_layout()
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
        # plt.plot(sample,[(sum(data['messgageNumOpt'])/len(data['messgageNumOpt']))/(sum(data['messgageNumZero'])/len(data['messgageNumZero'])) for x in range(0,len(sample))],'b',label='All blue',)
        plt.plot(sample,[1 for x in range(0,len(sample))],'r',lw=3,label='All red')
        

        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpLevel'],scaleFactor)
        plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='-.',marker='x',label='Level')
        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpMax'],scaleFactor)
        plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls=':',marker='s',label='Max')
        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpTop'],scaleFactor)
        plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='--',marker='^',label='Top')

        meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
        plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='--',marker='o',label='SMC')
        
    
        
      
        # plt.plot(sample,[data['messageExpLevel'][i]/scaleFactor[i] for i in range(0,n)],ls='-.',marker='x',label='Level')
        
        # plt.plot(sample,[data['messageExpMax'][i]/scaleFactor[i] for i in range(0,n)],ls=':',marker='s',label='Max')
        
    
        # plt.plot(sample,[data['messageExpTop'][i]/scaleFactor[i] for i in range(0,n)],ls='--',marker='^',label='Top')
        
        # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.set_xlim(1, 24)
        plt.grid(True, which="both", ls="-")
        plt.xlabel(r"Switch capacity",fontsize=18)
        plt.ylabel(r'Normalized Congestion',fontsize=18)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=3)
        
        plt.legend(loc='upper right',fontsize=14,ncol=2)
        # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        plt.savefig("Wieghted_Util_Multi_wieght_"+wieght+"_count_distribution_"+disrebution+"_scale_allRed_SOAR_bold.pdf",bbox_inches='tight')
        # plt.savefig("Weighted_Util_power_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
        # plt.close(2)
        # plt.figure(2)
        # ax = plt.gca()
        # M=sum(data['byteCountZero'])/len(data['byteCountZero'])
        # ax.set_box_aspect(1/2)
        # plt.tight_layout()
        # # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # meanAlg,varAlg=CalcMeanVar(data['bytsExpAlg'])
        # plt.errorbar(sample, [x/M for x in meanAlg], yerr=[x/M for x in varAlg],ls='--',marker='o',label='SOAR')
        
        # meanLevel,varLevel=CalcMeanVar(data['bytsExpLevel'])
        # plt.errorbar(sample,[x/M for x in meanLevel], yerr=[x/M for x in varLevel],ls='-.',marker='x',label='Level')
        
        # meanMax,varMax=CalcMeanVar(data['bytsExpMax'])
        # plt.errorbar(sample, [x/M for x in meanMax], yerr=[x/M for x in varMax],ls=':',marker='s',label='Max')
        
        # meanTop,varTop=CalcMeanVar(data['bytsExpTop'])
        # plt.errorbar([1,3,7,15,31,63],[x/M for x in meanTop], yerr=[x/M for x in varTop],ls='--',marker='^',label='Top')
        
        # # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.grid(True, which="both", ls="-")
        # plt.xlabel(r"Number of blue nodes ($\mathit{k})$",fontsize=15)
        # plt.ylabel('Bytes',fontsize=15)
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountOpt'])/len(data['byteCountOpt'])/M for x in range(0,len(sample))],'b',label='All blue')
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountZero'])/len(data['byteCountZero'])/M for x in range(0,len(sample))],'r',label='All red')
        # plt.legend(loc='upper right',fontsize=10)
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # # plt.title("Byts count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        # plt.savefig("WC_Byts_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')
    
def MutliJobMultiCapAlg_plots(dstDir,path,distrebutions,wieght):
    os.chdir('.\\'+dstDir)
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap_load_power_distrebutionRuns_10_tree_256\\"
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap_load_uni_Coin_distrebutionRuns_10_tree_256"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_linear_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_Power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\Multi_load_wieght_1_distrebutionRuns_tree_256\\"
    
    
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_FixedCap32_load_uni_distrebutionRuns_10_tree_256\\"
    # file=open(path+'\\Results.txt')
    # res=ast.literal_eval(file.read())
    # distrebutions=['Mixed','Uniform1','Uniform2','PowerLaw1']
    # distrebutions=['Coin']
    dec={'power':{'marker': '<','line':'--'},
         'linear':{'marker': '*','line':':'},
         'uniform':{'marker': 'h','line':'-.'}}
    plt.close(1)
    plt.figure(1)
    sample=[4,8,16,32]
    plt.plot(sample,[1 for x in range(0,len(sample))],'r',lw=3,label='All red')
    for w in wieght:
        for disrebution in distrebutions:
            directory_contents=os.listdir(path+'Multi_MultiCapT32_load_'+str(w)+'_distrebutionRuns_10_tree_256\\'+disrebution)
            data={}
            for f in directory_contents:
                try:
                    if f.split('.')[1] == 'txt':
                        file=open(path+'Multi_MultiCapT32_load_'+str(w)+'_distrebutionRuns_10_tree_256\\'+disrebution+'\\'+f)
                        key=f.split('.')[0]
                        data[key]=[]
                        lines=file.readlines()
                        for line in lines:
                            data[key].append(ast.literal_eval(line))
                            
                except:
                    print()
            
            M=np.power(10,6)
            # scaleFactor=[data['BnCountZero'][0][i] for i in range(1,len(data['BnCountZero'][0])+1)]
            # sample=[4,8,16]#[i for i in range(1,len(data['messageExpMax'])+1)]
           
            # n=len(data['messgageNumZero'])
            scaleFactor=data['BnCountZero']
            
            ax = plt.gca()
            ax.set_box_aspect(1/2)
            plt.tight_layout()
            # ax.set_ylim(ymin=0)
            # ax.set_xlim(xmin=0)
            # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
            # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
            # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
            # plt.plot(sample,[(sum(data['messgageNumOpt'])/len(data['messgageNumOpt']))/(sum(data['messgageNumZero'])/len(data['messgageNumZero'])) for x in range(0,len(sample))],'b',label='All blue',)
            
            meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls=dec[w]['line'],markersize=12,marker=dec[w]['marker'],label=w)

        
        # meanAlg,varAlg=CalcMeanVarCap(data['BnExpLevel'],scaleFactor)
        # plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='-.',marker='x',label='Level')
        
        # meanAlg,varAlg=CalcMeanVarCap(data['BnExpMax'],scaleFactor)
        # plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls=':',marker='s',label='Max')
        
        # meanAlg,varAlg=CalcMeanVarCap(data['BnExpTop'],scaleFactor)
        # plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='--',marker='^',label='Top')

        # meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
        # plt.errorbar(sample, [meanAlg[i] for i in range(0,len(sample))], yerr=[varAlg[i] for i in range(0,len(sample))],lw=3,markersize=12,ls='--',marker='o',label='SMC')
        
    
        
      
        # plt.plot(sample,[data['messageExpLevel'][i]/scaleFactor[i] for i in range(0,n)],ls='-.',marker='x',label='Level')
        
        # plt.plot(sample,[data['messageExpMax'][i]/scaleFactor[i] for i in range(0,n)],ls=':',marker='s',label='Max')
        
    
        # plt.plot(sample,[data['messageExpTop'][i]/scaleFactor[i] for i in range(0,n)],ls='--',marker='^',label='Top')
        
        # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.set_xlim(1, 24)
    # ax.set_yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.xlabel(r"Switch capacity",fontsize=18)
    plt.ylabel(r'Normalized Congestion',fontsize=18)
    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=3)
    
    plt.legend(loc='upper right',fontsize=14,ncol=2)
        # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
    plt.savefig("Wieghted_Util_Multi_Opt_count_distribution_"+disrebution+"_scale_allRed_SOAR_bold.pdf",bbox_inches='tight')
        # plt.savefig("Weighted_Util_power_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
        # plt.close(2)
        # plt.figure(2)
        # ax = plt.gca()
        # M=sum(data['byteCountZero'])/len(data['byteCountZero'])
        # ax.set_box_aspect(1/2)
        # plt.tight_layout()
        # # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # meanAlg,varAlg=CalcMeanVar(data['bytsExpAlg'])
        # plt.errorbar(sample, [x/M for x in meanAlg], yerr=[x/M for x in varAlg],ls='--',marker='o',label='SOAR')
        
        # meanLevel,varLevel=CalcMeanVar(data['bytsExpLevel'])
        # plt.errorbar(sample,[x/M for x in meanLevel], yerr=[x/M for x in varLevel],ls='-.',marker='x',label='Level')
        
        # meanMax,varMax=CalcMeanVar(data['bytsExpMax'])
        # plt.errorbar(sample, [x/M for x in meanMax], yerr=[x/M for x in varMax],ls=':',marker='s',label='Max')
        
        # meanTop,varTop=CalcMeanVar(data['bytsExpTop'])
        # plt.errorbar([1,3,7,15,31,63],[x/M for x in meanTop], yerr=[x/M for x in varTop],ls='--',marker='^',label='Top')
        
        # # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.grid(True, which="both", ls="-")
        # plt.xlabel(r"Number of blue nodes ($\mathit{k})$",fontsize=15)
        # plt.ylabel('Bytes',fontsize=15)
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountOpt'])/len(data['byteCountOpt'])/M for x in range(0,len(sample))],'b',label='All blue')
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountZero'])/len(data['byteCountZero'])/M for x in range(0,len(sample))],'r',label='All red')
        # plt.legend(loc='upper right',fontsize=10)
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # # plt.title("Byts count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        # plt.savefig("WC_Byts_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')

def MutliJobFixCapAlgVS_plots(dstDir,path,distrebutions,wieght):
    os.chdir('.\\'+str(dstDir))
    for disrebution in distrebutions:
        directory_contents=os.listdir(path+'\\'+disrebution)
        data={}
        for f in directory_contents:
            try:
                if f.split('.')[1] == 'txt':
                    file=open(path+'\\'+disrebution+'\\'+f)
                    key=f.split('.')[0]
                    data[key]=[]
                    lines=file.readlines()
                    for line in lines:
                        data[key].append(ast.literal_eval(line))
                        
            except:
                print()
        
        M=np.power(10,6)
        # scaleFactor=[data['BnCountZero'][0][i] for i in range(0,len(data['BnCountZero'][0]))]
        # sample=[4,8,16]#[i for i in range(1,len(data['messageExpMax'])+1)]
        scaleFactor=data['BnCountZero']
        sample=[1,2,4,8,16,32]
        # n=len(data['BnCounZero'])
        plt.close(1)
        plt.figure(1)
        ax = plt.gca()
        ax.set_box_aspect(1/2)
        plt.tight_layout()
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
        # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
        # plt.plot(sample,[(sum(data['messgageNumOpt'])/len(data['messgageNumOpt']))/(sum(data['messgageNumZero'])/len(data['messgageNumZero'])) for x in range(0,len(sample))],'b',label='All blue',)
        plt.plot(sample,[1 for x in range(0,len(sample))],'r',lw=3,label='All red')
        

        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpLevel'],data['BnCountZero'])
        plt.errorbar(sample, [meanAlg[x-1] for x in sample], yerr=[varAlg[x-1] for x in sample],lw=3,ls='-.',markersize=12,marker='x',label='Level')
        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpMax'],data['BnCountZero'])
        plt.errorbar(sample, [meanAlg[x-1] for x in sample], yerr=[varAlg[x-1] for x in sample],lw=3,ls=':',markersize=12,marker='s',label='Max')
        
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpTop'],data['BnCountZero'])
        plt.errorbar(sample, [meanAlg[x-1] for x in sample], yerr=[varAlg[x-1] for x in sample],lw=3,ls='--',markersize=12,marker='^',label='Top')
        
    
        meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],data['BnCountZero'])
        plt.errorbar(sample, [meanAlg[x-1] for x in sample], yerr=[varAlg[x-1] for x in sample],color='tab:purple',lw=3,ls='--',markersize=12,marker='o',label='SMC')
      
        # plt.plot(sample,[data['messageExpLevel'][i]/scaleFactor[i] for i in range(0,n)],ls='-.',marker='x',label='Level')
        
        # plt.plot(sample,[data['messageExpMax'][i]/scaleFactor[i] for i in range(0,n)],ls=':',marker='s',label='Max')
        
    
        # plt.plot(sample,[data['messageExpTop'][i]/scaleFactor[i] for i in range(0,n)],ls='--',marker='^',label='Top')
        
        # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.set_xlim(1, 24)
        plt.grid(True, which="both", ls="-")
        plt.xlabel(r"Number Workloads",fontsize=18)
        plt.ylabel(r'Normalized Congestion',fontsize=18)
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        
        plt.legend(loc='upper right',fontsize=14,ncol=2)
        # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        plt.savefig("BN_fixCap_wieght_"+wieght+"_count_distribution_"+disrebution+"_scale_allRed_SOAR_bold.pdf",bbox_inches='tight')
        # plt.savefig("Weighted_Util_power_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
        # plt.close(2)
        # plt.figure(2)
        # ax = plt.gca()
        # M=sum(data['byteCountZero'])/len(data['byteCountZero'])
        # ax.set_box_aspect(1/2)
        # plt.tight_layout()
        # # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
        # meanAlg,varAlg=CalcMeanVar(data['bytsExpAlg'])
        # plt.errorbar(sample, [x/M for x in meanAlg], yerr=[x/M for x in varAlg],ls='--',marker='o',label='SOAR')
        
        # meanLevel,varLevel=CalcMeanVar(data['bytsExpLevel'])
        # plt.errorbar(sample,[x/M for x in meanLevel], yerr=[x/M for x in varLevel],ls='-.',marker='x',label='Level')
        
        # meanMax,varMax=CalcMeanVar(data['bytsExpMax'])
        # plt.errorbar(sample, [x/M for x in meanMax], yerr=[x/M for x in varMax],ls=':',marker='s',label='Max')
        
        # meanTop,varTop=CalcMeanVar(data['bytsExpTop'])
        # plt.errorbar([1,3,7,15,31,63],[x/M for x in meanTop], yerr=[x/M for x in varTop],ls='--',marker='^',label='Top')
        
        # # plt.plot(sample,messageExpLevel[0],'-',marker='x',label='Level')
        # # plt.plot(sample,messageExpMax[0],'-.',marker='s',label='Max')
        # # plt.plot([1,3,7,15,31],messageExpTop[0],'--',marker='^',label='Top')
        # plt.grid(True, which="both", ls="-")
        # plt.xlabel(r"Number of blue nodes ($\mathit{k})$",fontsize=15)
        # plt.ylabel('Bytes',fontsize=15)
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountOpt'])/len(data['byteCountOpt'])/M for x in range(0,len(sample))],'b',label='All blue')
        # plt.plot([0,1,3,7,15,31,65],[sum(data['byteCountZero'])/len(data['byteCountZero'])/M for x in range(0,len(sample))],'r',label='All red')
        # plt.legend(loc='upper right',fontsize=10)
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(xmin=0)
        # # plt.title("Byts count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
        # plt.savefig("WC_Byts_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')


def AlgOpt_plots(dstDir,path,distrebutions,wieght,scale='allRed'):
    if not(os.path.isdir(dstDir)):
        os.mkdir(dstDir)
    os.chdir('.//'+dstDir)
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_linear_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\distrebutionRuns_paremeterServer_range_10000_lenght_5000_new3\\"
    # file=open(path+'\\Results.txt')
    # res=ast.literal_eval(file.read())
    # distrebutions=['Mixed','Uniform1','Uniform2','Skewed','PowerLaw1','PowerLaw3','PowerLaw2']
    distrebutions=['Uniform2','PowerLaw1']
    plt.close(1)
    plt.figure(1)
    data={}
    for disrebution in distrebutions:
        directory_contents=os.listdir(path+'\\'+disrebution)
        data[disrebution]={}
        for f in directory_contents:
            try:
                if f.split('.')[1] == 'txt':
                    file=open(path+'\\'+disrebution+'\\'+f)
                    key=f.split('.')[0]
                    data[disrebution][key]=[]
                    lines=file.readlines()
                    for line in lines:
                        data[disrebution][key].append(ast.literal_eval(line))
                        
            except:
                print()
                
    sample=[1,2,4,8,16,32]
    s=[]    
    for i in data['Uniform2']['BnCountZero']:
        st=[]
        for j in sample:
            st.append(1)
        s.append(st)
    scaleFactor=s
    M=np.power(10,6)
    
    scaleFactor=1#sum(data['messgageNumZero'])/len(data['messgageNumZero'])
    if scale == 'alg':
        scaleFactor=data['BnExpAlg']
    else:
        scaleFactor=s

       
    sF=scaleFactor

    ax = plt.gca()
    ax.set_box_aspect(1/2)
    plt.tight_layout()
    # ax.set_ylim(ymin=0)
    # ax.set_xlim(xmin=0)
    # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
    # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
    # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
    # plt.plot([0,1,3,7,15,32],[(sum(data['BnCountOpt'])/len(data['BnCountOpt']))/(sum(data['BnCountZero'])/len(data['BnCountZero'])) for x in range(0,len(sample))],'b',lw=3,label='All blue',)
    # plt.plot([0,1,3,7,15,32],[1 for x in range(0,len(sample))],'r',lw=3,label='All red')

    # if scale == 'alg':
    #     s=[]    
    #     for x in scaleFactor:
    #         s.append([x[i] for i in [0,1,3,5,7,9]])
    #     sF=s
    # else:
    #     sF=scaleFactor
    # meanLevel,varLevel=CalcMeanVarCap(data['BnExpLevel'],sF)
    # plt.errorbar(sample, [x for x in meanLevel], yerr=[x for x in varLevel],lw=3,ls='-.',markersize=12,marker='x',label='Level')
    
    # meanMax,varMax=CalcMeanVarCap(data['BnExpMax'],sF)
    # plt.errorbar(sample, [x for x in meanMax], yerr=[x for x in varMax],ls=':',lw=3,marker='s',markersize=12,label='Max')
    # if scale == 'alg':
    #     s=[]    
    #     for x in scaleFactor:
    #         s.append([x[i] for i in [0,1,3,5,7,9]])
    #     sF=s
    # else:
    #     sF=scaleFactor
    # meanTop,varTop=CalcMeanVarCap(data['BnExpTop'],sF)
    # plt.errorbar(sample, [x for x in meanTop], yerr=[x for x in varTop],lw=3,ls='--',marker='^',markersize=12,label='Top')


    # meanAlg,varAlg=CalcMeanVarCap(data['BnCountZero'],scaleFactor)
    # plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls='--',color='r',markersize=12,marker='-',label='All red')
    # plt.plot(sample,BnExpLevel[0],'-',marker='x',label='Level')
    # plt.plot(sample,BnExpMax[0],'-.',marker='s',label='Max')
    # plt.plot([1,3,7,15,31],BnExpTop[0],'--',marker='^',label='Top')
    dec={'Uniform2':{'marker': 'd','line':'--','color':'tab:purple'},
         'PowerLaw1':{'marker': 'p','line':':','color':'tab:purple'}}
    plt.grid(True, which="both", ls="-")
    plt.xlabel(r"Number of blue nodes ($k$)",fontsize=18)
    plt.ylabel(r'Network Congestion',fontsize=18)
    for disrebution in distrebutions:
        zero=[]    
        for i in data[disrebution]['BnCountZero']:
                st=[]
                for j in sample:
                    st.append(i)
                zero.append(st)
    
        if scale == 'alg':
            Loc='upper left'
            meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],color='b',lw=3,ls='-',markersize=12,label='SMC')
            meanAlg,varAlg=CalcMeanVarCap(s,scaleFactor)
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,color='r',ls=':',markersize=12,label='All red')
            ax.set_yscale('log',base=2)
            ax.set_ylim(0,16)
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
        else:
            Loc='upper right'
            
            # ax.set_ylim(ymin=0)
            
            meanAlg,varAlg=CalcMeanVarCap(data[disrebution]['BnExpAlg'],sF)
            name=disrebution.replace("1","")
            name=name.replace("2","")
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,color=dec[disrebution]['color'],ls=dec[disrebution]['line'],markersize=12,marker=dec[disrebution]['marker'],label=name)
            meanAlg,varAlg=CalcMeanVarCap(zero,sF)
            name=disrebution.replace("1","")
            name=name.replace("2","")
            plt.errorbar(sample, [x for x in meanAlg], yerr=[0 for x in varAlg],lw=3,color='r',ls=dec[disrebution]['line'],marker=dec[disrebution]['marker'],mfc='white',markersize=12,label=name+'- all red')
    plt.plot(sample,[1 for x in range(0,len(sample))],'b',lw=3,label='All Blue')
    # ax.set_yscale('log')
    # ax.set_ylim(1)
    if weight == 'uniform':
        ax.set_ylim(0,400)
    else:
        ax.set_ylim(0,70)
    
        # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    ax.set_xlim(xmin=0)
   
    
    
    plt.legend(loc=Loc,fontsize=12,ncol=1)
    # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
    # plt.savefig("Weighted_Util_Multi_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
    plt.savefig("Weighted_Util_"+wieght+"_count_distribution_"+disrebution+"_scale_"+scale+"_SOAR_bold.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')

def AlgOptWieghts_plots(dstDir,path,distrebutions,wieght,scale='allRed'):
    if not(os.path.isdir(dstDir)):
        os.mkdir(dstDir)
    os.chdir('.//'+dstDir)
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_linear_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\Multi_load_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\wietghed_power_distrebutionRuns_5_tree_256\\"
    # path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\distrebutionRuns_paremeterServer_range_10000_lenght_5000_new3\\"
    # file=open(path+'\\Results.txt')
    # res=ast.literal_eval(file.read())
    # distrebutions=['Mixed','Uniform1','Uniform2','Skewed','PowerLaw1','PowerLaw3','PowerLaw2']
    distrebutions=['Uniform2','PowerLaw1']
    # name='Bottleneck_wietghed_'+w+'_distrebutionRuns_10_tree_256\\
    data={}
    for w in wieght:
        data[w]={}
        for disrebution in distrebutions:
            directory_contents=os.listdir(path+'Bottleneck_wietghed_'+w+'_distrebutionRuns_10_tree_256\\'+disrebution)
            data[w][disrebution]={}
            for f in directory_contents:
                try:
                    if f.split('.')[1] == 'txt':
                        file=open(path+'Bottleneck_wietghed_'+w+'_distrebutionRuns_10_tree_256\\'+disrebution+'\\'+f)
                        key=f.split('.')[0]
                        data[w][disrebution][key]=[]
                        lines=file.readlines()
                        for line in lines:
                            data[w][disrebution][key].append(ast.literal_eval(line))
                            
                except:
                    print()
                
    sample=[1,2,4,8,16,32]
    s=[]    
    for i in data['power']['Uniform2']['BnCountZero']:
        st=[]
        for j in sample:
            st.append(1)
        s.append(st)
    scaleFactor=s
    M=np.power(10,6)
    
    scaleFactor=1#sum(data['messgageNumZero'])/len(data['messgageNumZero'])
    if scale == 'alg':
        scaleFactor=data['BnExpAlg']
    else:
        scaleFactor=s

    
    
    sF=scaleFactor

    dec={'power':{'marker': 'o','line':'--'},
         'linear':{'marker': 's','line':':'},
         'uniform':{'marker': 'x','line':'-.'}}
    for disrebution in distrebutions:
        plt.close(1)
        plt.figure(1)
        for w in wieght:
            meanAlg,varAlg=CalcMeanVarCap(data[w][disrebution]['BnExpAlg'],sF)
            name=disrebution.replace("1","")
            name=name.replace("2","")
            plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls=dec[w]['line'],markersize=12,marker=dec[w]['marker'],label=w)
        plt.plot(sample,[1 for x in range(0,len(sample))],'b',lw=3,label='All Blue')    
        ax = plt.gca()
        # ax.set_yscale('log')
        ax.set_box_aspect(1/2)
        plt.tight_layout()
        plt.legend(fontsize=12,ncol=1)
        plt.grid(True, which="both", ls="-")
        # ax.set_yscale('log')
        ax.set_ylim(ymin=0)
        ax.set_xlim(xmin=0)
        plt.xlabel(r"Number of blue nodes ($k$)",fontsize=18)
        plt.ylabel(r'Normalized Congestion',fontsize=18)
        plt.savefig("Weighted_Util_Opt_count_distribution_"+disrebution+"_scale_"+scale+"_SOAR_bold.pdf",bbox_inches='tight')

    # plt.plot(sample,messageExpAlg[0],'--',marker='o',label='Truffel')
    # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumOpt'])/len(data['messgageNumOpt'])/scaleFactor for x in range(0,len(sample))],'b',label='All blue',)
    # plt.plot([0,1,3,7,15,31],[sum(data['messgageNumZero'])/len(data['messgageNumZero'])/scaleFactor for x in range(0,len(sample))],'r',label='All red')
    # plt.plot([0,1,3,7,15,32],[(sum(data['BnCountOpt'])/len(data['BnCountOpt']))/(sum(data['BnCountZero'])/len(data['BnCountZero'])) for x in range(0,len(sample))],'b',lw=3,label='All blue',)
    # plt.plot([0,1,3,7,15,32],[1 for x in range(0,len(sample))],'r',lw=3,label='All red')

    # if scale == 'alg':
    #     s=[]    
    #     for x in scaleFactor:
    #         s.append([x[i] for i in [0,1,3,5,7,9]])
    #     sF=s
    # else:
    #     sF=scaleFactor
    # meanLevel,varLevel=CalcMeanVarCap(data['BnExpLevel'],sF)
    # plt.errorbar(sample, [x for x in meanLevel], yerr=[x for x in varLevel],lw=3,ls='-.',markersize=12,marker='x',label='Level')
    
    # meanMax,varMax=CalcMeanVarCap(data['BnExpMax'],sF)
    # plt.errorbar(sample, [x for x in meanMax], yerr=[x for x in varMax],ls=':',lw=3,marker='s',markersize=12,label='Max')
    # if scale == 'alg':
    #     s=[]    
    #     for x in scaleFactor:
    #         s.append([x[i] for i in [0,1,3,5,7,9]])
    #     sF=s
    # else:
    #     sF=scaleFactor
    # meanTop,varTop=CalcMeanVarCap(data['BnExpTop'],sF)
    # plt.errorbar(sample, [x for x in meanTop], yerr=[x for x in varTop],lw=3,ls='--',marker='^',markersize=12,label='Top')


    # meanAlg,varAlg=CalcMeanVarCap(data['BnCountZero'],scaleFactor)
    # plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls='--',color='r',markersize=12,marker='-',label='All red')
    # plt.plot(sample,BnExpLevel[0],'-',marker='x',label='Level')
    # plt.plot(sample,BnExpMax[0],'-.',marker='s',label='Max')
    # plt.plot([1,3,7,15,31],BnExpTop[0],'--',marker='^',label='Top')
    # dec={'Uniform2':{'marker': 'o','line':'--'},
    #      'PowerLaw1':{'marker': 's','line':':'}}
    # plt.grid(True, which="both", ls="-")
    # plt.xlabel(r"Number of blue nodes ($k$)",fontsize=18)
    # plt.ylabel(r'Normalized Congestion',fontsize=18)
    # for disrebution in distrebutions:
    #     zero=[]    
    #     for i in data[disrebution]['BnCountZero']:
    #             st=[]
    #             for j in sample:
    #                 st.append(i)
    #             zero.append(st)
    
    #     if scale == 'alg':
    #         Loc='upper left'
    #         meanAlg,varAlg=CalcMeanVarCap(data['BnExpAlg'],scaleFactor)
    #         plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],color='b',lw=3,ls='-',markersize=12,label='SMC')
    #         meanAlg,varAlg=CalcMeanVarCap(s,scaleFactor)
    #         plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,color='r',ls=':',markersize=12,label='All red')
    #         ax.set_yscale('log',base=2)
    #         ax.set_ylim(0,16)
    #         ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    #     else:
    #         Loc='upper right'
            
    #         # ax.set_ylim(ymin=0)
            
    #         meanAlg,varAlg=CalcMeanVarCap(data[disrebution]['BnExpAlg'],sF)
    #         name=disrebution.replace("1","")
    #         name=name.replace("2","")
    #         plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,ls=dec[disrebution]['line'],markersize=12,marker=dec[disrebution]['marker'],label=name)
    #         meanAlg,varAlg=CalcMeanVarCap(zero,sF)
    #         name=disrebution.replace("1","")
    #         name=name.replace("2","")
    #         plt.errorbar(sample, [x for x in meanAlg], yerr=[x for x in varAlg],lw=3,color='r',ls=dec[disrebution]['line'],markersize=12,label=name+'- all red')
    # plt.plot(sample,[1 for x in range(0,len(sample))],'b',lw=3,label='All Blue')
    # # ax.set_yscale('log')
    # ax.set_ylim(0)
    
    #     # ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    # ax.set_xlim(xmin=0)
   
    
    
    # plt.legend(loc=Loc,fontsize=12,ncol=1)
    # # plt.title("Message count- distribution: "+str(d[0])+", "+str(len(messageExpAlg))+" experments")
    # # plt.savefig("Weighted_Util_Multi_count_distribution_"+disrebution+"_scale_allRed_SOAR.pdf",bbox_inches='tight')
    # plt.savefig("Weighted_Util_"+wieght+"_count_distribution_"+disrebution+"_scale_"+scale+"_SOAR_bold.pdf",bbox_inches='tight')
    os.chdir('..')
    os.chdir('..')

# weight='uniform'
expNum=10

dstDir='InfoCom_figs'
path="C:\\Users\\Segal Raz\\OneDrive - post.bgu.ac.il\\Documents\\master\\"
distrebutions=['Mixed','Uniform1','Uniform2','Skewed','PowerLaw1','PowerLaw3','PowerLaw2']
# distrebutions=['PowerLaw2']
weights=['uniform','power','linear']
# weights=['power']
for weight in weights:
    # AlgVS(weight,expNum)
    name='Bottleneck_wietghed_'+weight+'_distrebutionRuns_'+str(expNum)+'_tree_256\\'
    # AlgVS_plots(dstDir+'\\AlgVs',path+name,distrebutions,weight,'allRed')
    AlgVS_plots(dstDir+'\\AlgVs',path+name,distrebutions,weight,'alg') 
    AlgOpt_plots(dstDir+'\\AlgOpt',path+name,distrebutions,weight,'opt')
# AlgOptWieghts_plots(dstDir+'\\AlgOpt',path,distrebutions,weights,scale='allRed')
cap=4

distrebutions=['Coin']
for weight in weights:
    # MutliJobFixCapAlgVS(weight,expNum,cap)
    name='Multi_FixedCapT'+str(cap)+'_load_'+str(weight)+'_distrebutionRuns_'+str(expNum)+'_tree_256\\'
    MutliJobFixCapAlgVS_plots(dstDir+'\\MultiloadFixedCap',path+name,distrebutions,weight) 
caps=[4,8,16,32]
for weight in weights:
    # MultiJobsMultiCap(weight,expNum,caps)
    name='Multi_MultiCapT'+str(max(caps))+'_load_'+str(weight)+'_distrebutionRuns_'+str(expNum)+'_tree_256\\'
    # MutliJobMultiCapAlgVS_plots(dstDir+'\\MultiloadMultiCap',path+name,distrebutions,weight) 

MutliJobMultiCapAlg_plots(dstDir+'\\AlgOpt',path,distrebutions,weights)
# X=7 
# k=1
# # g=nx.read_adjlist("Test_Tree.txt",create_using=nx.DiGraph,nodetype=int)
# deg=2
# h=7
# distrbution=paseDistrebution('distrebutions.txt')
# cap=1
# dis=distrbution[5]
# # for i in range(0,50):
# g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
# # loadDist=list(np.random.permutation([1,1,2,2,2,2,3,3]))
# loadDist=list(np.random.permutation(dis[1]))
# root=0
# leafL=leafList(g)
# addLoad(g,loadDist,leafL)
# Add_InNetwork_Capacity(g)
# AddWieghtToEges(g,root,"linear")
# # x=findXnew(g,root,k,cap)
# Avilabilty=AvalbiltyCalc(g,1)
# X=findX(g,root,k,Avilablity)
# print('X: '+str(X))
# # Walg.run(g,root,k)
# # g=plot_coloring(g,[11,12,13,14])
# # Walg.NewColoring(g,root,root,0,k)
# # GtoFile(g,"Eample_X_"+str(X))
# Avilabilty=AvalbiltyCalc(g,cap)
# BNalg.run(g,root,k,X,Avilabilty)
# BNalg.NewColoring(g,root,root,k,X)
# Walg.messageCount(g,root)
# SumMessage,BottleNeck=NetworkUtiliztion(g)
# # BNalg.run(g,root,k,82.8,Avilabilty)
# # if round(SumMessage,8) != round(g.nodes[root]['minSend']['l0']['k'+str(k)],8):
# #     print("Error")
#     # break

# # # BootleNALG[dis[0]]['SumMessage']=SumMessage
# # # BootleNALG[dis[0]]['BottleNeck']=BottleNeck
# plt.figure(0)
# plt.title('Toy example,BN alg, number of Utilization: '+str(SumMessage)+' ,bottle neck: '+str(BottleNeck))
# labels = nx.get_node_attributes(g, 'load') 
# # nx.draw(g,pos=graphviz_layout(g, prog="dot"),with_labels=True)
# nx.draw(g,pos=graphviz_layout(g, prog="dot"),labels=labels)
# nx.draw(g,pos=graphviz_layout(g, prog='dot'),node_color=colorMap(g),labels=labels)
# edge_labels={}
# for e in g.edges:
#     edge_labels[e]={'e':e,'w':g.edges[e]['Wieght'],'m':g.edges[e]['mesageCount']}
# edge_labels = nx.get_edge_attributes(g,'Wieght')
# pos=graphviz_layout(g,prog='dot')
# nx.draw_networkx_edge_labels(g, pos, edge_labels = edge_labels,rotate=False)

# gr=plot_coloring(g,MaxColor(k, g, leafL))
# Walg.messageCount(gr,root)
# SumMessage,BottleNeck=NetworkUtiliztion(gr)

# plt.figure(1)
# plt.title('max, number of Utilization: '+str(SumMessage)+' ,bottle neck: '+str(BottleNeck))
# labels = nx.get_node_attributes(gr, 'load') 
# nx.draw(gr,pos=graphviz_layout(gr, prog="dot"),node_color=colorMap(gr),with_labels=True)
# # nx.draw(g,pos=graphviz_layout(g, prog="dot"),labels=labels)
# # nx.draw(g,pos=graphviz_layout(g, prog='dot'),node_color=colorMap(g),labels=labels)
# edge_labels={}
# for e in g.edges:
#     edge_labels[e]={'w':gr.edges[e]['Wieght'],'m':gr.edges[e]['mesageCount']}
# # edge_labels = nx.get_edge_attributes(g,'Wieght')
# pos=graphviz_layout(gr,prog='dot')
# nx.draw_networkx_edge_labels(gr, pos, edge_labels = edge_labels,rotate=True)

