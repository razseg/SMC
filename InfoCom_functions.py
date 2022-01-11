# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:18:12 2021

@author: Segal Raz
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 23:54:59 2021

@author: Segal Raz
"""
import networkx as nx
import matplotlib.pyplot as plt
import threading
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.ticker as mticker
import math
import os
import numpy as np
import copy
import time

def phi(g,node,l,i):
    if i<0:
        return math.inf
    if i==0:
        return g.nodes[node]['l'+str(l)]['k'+str(i)]['red']['val']
    return min(g.nodes[node]['l'+str(l)]['k'+str(i)]['red']['val'],g.nodes[node]['l'+str(l)]['k'+str(i)]['blue']['val'])

def bestPartition(g,node, neighbour, l, i,col):
    tmp=[]
    
    for j in range(0,i+1):
        #print('phi'+str(j)+':'+str(phi(g,neighbour, 1 if col=='blue' else (l+1),j)))
        #print('nphi '+str(i-j)+' :' +str(g.nodes[node]['l'+str(l)]['k'+str(i-j)][col]['val']))
        tmp.append(g.nodes[node]['l'+str(l)]['k'+str()][col]['val']+phi(g,neighbour, 1 if col=='blue' else (l+1),i-j))
    #print("bestPartition: "+str(tmp))
    return [min(tmp),np.argmin(tmp)]

def nodeLoad(g,node):
    neighborsList= list(g.neighbors(node))
    load=0
    workers=[]
    for n in neighborsList:
        if g.nodes[n]['type']=='w':
            load=load+1
            workers.append(n)
    return [load,workers]

def mCost(i,PreviosChild,Child,X,rate,color):
    tmp=[]
    # if color == 'blue':
    #     i=i-1
    if color =='red':
        for j in range(0,i+1):
            if PreviosChild['k'+str(i-j)]['bn']['red'] and Child['k'+str(j)]['bn']:
                tmp.append(PreviosChild['k'+str(i-j)]['U']['red']+Child['k'+str(j)]['U'])
        tmp.append(math.inf)
        U=min(tmp)
        if U*rate<=X:
            return [U,True]
        else:
            return [U,False]
    else:
        phi=False
        for j in range(0,i+1):
            phi= phi or (PreviosChild['k'+str(i-j)]['bn']['blue'] and Child['k'+str(j)]['bn'])
        return [1,phi]


def mSplit(i,PreviosChild,Child,X,color):
    tmp=[]
    # if color == 'blue':
    #     i=i-1
    if color =='red':
        for j in range(0,i+1):
            if PreviosChild['k'+str(i-j)]['bn']['red'] and Child['k'+str(j)]['bn']:
                tmp.append(PreviosChild['k'+str(i-j)]['U']['red']+Child['k'+str(j)]['U'])
            else:
                tmp.append(math.inf)
        return np.argmin(tmp)
    else:
       # phi=False
        for j in range(0,i+1):
            # phi= phi or (PreviosChild['k'+str(i-j)]['bn']['blue'] and Child['k'+str(j)]['bn'])
            if (PreviosChild['k'+str(i-j)]['bn']['blue'] and Child['k'+str(j)]['bn']):
                tmp.append(1)
            else:
                tmp.append(math.inf)
        return np.argmin(tmp)
    
    
def nodeRun(graph,node,root,k,X,Avilabilty):
        load=graph.nodes[node]['load'] #Note to take care
        n=[x for x in list(graph.neighbors(node))]
        #depth=nx.dijkstra_path_length(graph,root,node)
        att={ node : {'m'+str(c):{'k'+str(i):{'U':{'blue':0,'red':0},'bn':{'blue':False,'red':False}} for i in range(0,k+1)} for c in range(0,len(n)) }}
        att[node]['minSend']={'k'+str(i):{'U':0,'bn':False} for i in range(0,k+1) for c in range(0,len(n) if n else 1)  }
        att[node]['node']=node
        att[node]['children']=[]
        att[node]['color']='red'
        nx.set_node_attributes(graph,att)
        # print('node: '+str(node))
        # if node == 0:
        #     print('stop')
        rate=0
        try:
            parent=list(graph.predecessors(node))[0]
            rate=1/(graph.edges[(parent,node)]['Wieght'])
        except:
                pass
        if not n:
            # print("leaf: "+str(node)+" load:"+str(load))

            
            graph.nodes[node]['minSend']['k'+str(0)]['U']=load
            if (float(load*rate) <= X):
                graph.nodes[node]['minSend']['k'+str(0)]['bn']=True
            else:
                graph.nodes[node]['minSend']['k'+str(0)]['bn']=False
            for i in range(1,k+1):
                if Avilabilty[node]:
                    graph.nodes[node]['minSend']['k'+str(i)]['U']=1
                    graph.nodes[node]['minSend']['k'+str(i)]['bn']=True
                else:
                    graph.nodes[node]['minSend']['k'+str(i)]['U']=graph.nodes[node]['minSend']['k'+str(0)]['U']
                    graph.nodes[node]['minSend']['k'+str(i)]['bn']=graph.nodes[node]['minSend']['k'+str(0)]['bn']
                    
            # for i in range(0,k+1):
                # print('node: '+str(node)+', Sent,k: '+str(i)+ ' U: '+str(graph.nodes[node]['minSend']['k'+str(i)]['U'])+', bn: '+str(graph.nodes[node]['minSend']['k'+str(i)]['bn']))
                    
        else: 
            first=True  
            for neighbour in n:
                graph.nodes[node]['children'].append(neighbour)
                # print("(n.index(neighbour) "+str(n.index(neighbour)))
                if first:
                    for i in range(0,k+1):
                        graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['U']['red']=graph.nodes[neighbour]['minSend']['k'+str(i)]['U']+load
                        if graph.nodes[neighbour]['minSend']['k'+str(i)]['bn'] and (float(graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['U']['red']*rate)<= X):
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['red']=True
                        else:
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['red']=False
                        graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['U']['blue']=1
                        if i>0 and Avilabilty[node] and graph.nodes[neighbour]['minSend']['k'+str(i-1)]['bn']:
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['blue']=True
                        else:
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['blue']=False                            
                    first=False
                else:
                    # tmpNode=copy.deepcopy(graph.nodes[node])
                    PreviosChild=graph.nodes[node]['m'+str((n.index(neighbour)-1))]
                    Child=graph.nodes[neighbour]['minSend']
                    for i in range(0,k+1):
                        if Avilabilty[node]:
                            U,bn=mCost(i,PreviosChild,Child,X,rate,'blue')
                            # print('node: '+str(node)+', blue,k: '+str(i)+ ' U: '+str(U)+', bn: '+str(bn))
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['U']['blue']=U
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['blue']=bn
                        else:
                            graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['blue']=False
                        U,bn=mCost(i,PreviosChild,Child,X,rate,'red')
                        # print('node: '+str(node)+', red,k: '+str(i)+' U: '+str(U)+', bn: '+str(bn))
                        graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['U']['red']=U
                        graph.nodes[node]['m'+str(n.index(neighbour))]['k'+str(i)]['bn']['red']=bn

                            
                            
                    # att={node: tmpNode}
                    # nx.set_node_attributes(graph,att)
            neihbourNum=len(n)-1
            for i in range(0,k+1):
                if graph.nodes[node]['m'+str(neihbourNum)]['k'+str(i)]['bn']['blue']:
                    graph.nodes[node]['minSend']['k'+str(i)]['U']=1
                    graph.nodes[node]['minSend']['k'+str(i)]['bn']=True
                else:
                    graph.nodes[node]['minSend']['k'+str(i)]['U']=graph.nodes[node]['m'+str(neihbourNum)]['k'+str(i)]['U']['red']
                    graph.nodes[node]['minSend']['k'+str(i)]['bn']=graph.nodes[node]['m'+str(neihbourNum)]['k'+str(i)]['bn']['red']
                # print('node: '+str(node)+', Sent,k: '+str(i)+ ' U: '+str(graph.nodes[node]['minSend']['k'+str(i)]['U'])+', bn: '+str(graph.nodes[node]['minSend']['k'+str(i)]['bn']))
                    
def nodeThread(*args):
    # print("node " +str(args[1])+' thread')
    # print(args)
    nodeRun(args[0],args[1],args[2],args[3])
    # print("node " +str(node)+' thread ended')

def threadrun(g,root,deg,h,k):
    start_time = time.time()
    
    
    l=list (nx.all_pairs_dijkstra_path_length(g))
    rootIndex=list(g.nodes).index(root)
    depth=max(l[rootIndex][1].values())
    
    r={}
    for i in range(0,depth+1):
        r[i]=[]
    for node in g.nodes():
         r[l[rootIndex][1][node]].append(node)

    for i in range(0,depth+1):
        threads = []
        level_time = time.time()
        print('deg:'+str(deg)+'i: '+str(i))
        for node in r[depth-i]: 
            # print(node)
            if g.nodes[node]['type'] == 's':
                #print("node call"+str(node))
                t = threading.Thread(target=nodeThread, args=(g,node,root,k,))
                threads.append(t)
                t.start()
            for x in threads:
                x.join()
        # print ("round " +str(i)+' finished')
        print("level time: "+str(time.time()-level_time))
        
    print("Running time: "+str(time.time()-start_time))
    return g

def run(g,root,k,X,Avilabilty):
    start_time = time.time()
    # g=nx.balanced_tree(deg,h,create_using=nx.DiGraph)
    
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
                nodeRun(g,node,root,k,X,Avilabilty)
    
    # print("Running time: "+str(time.time()-start_time))
    return g

# def coloring(gr,node,root,k,d):
#     # K=k
#     if node==root:
#         for n in gr.nodes:
#             if gr.nodes[n]['type'] == 's':
#                 gr.nodes[n]['color']='red'
#             else:
#                  gr.nodes[n]['color']='gray'
#     # if((gr.nodes[node]['tree']['blue'][k-1] if (k-1)>=0 else math.inf) <gr.nodes[node]['tree']['red'][k]):
#     #     gr.nodes[node]['color']='blue'
#     children=gr.nodes[node]['children']
#     if children:
#         if gr.nodes[node]['m'+str(len(children)-1)]['l'+str(d)]['k'+str(k)]['blue']<gr.nodes[node]['m'+str(len(children)-1)]['l'+str(d)]['k'+str(k)]['red'] and k>0:
#             gr.nodes[node]['color']='blue'
        
        

#     color=gr.nodes[node]['color']
    
#     for c in range(0,len(children)):

#         # col=childColor(color,gr.nodes[children[len(children)-c-1]]['tree'],gr.nodes[children[len(children)-c-1]]['redSent'],i)
#         if k>0:
#             child=children[len(children)-c-1]
#             PreviosYm=gr.nodes[node]['m'+str(len(children)-c-1)]
#             Xm=gr.nodes[child]['minSend']
#             i=mSplit(gr,d,k,PreviosYm,Xm,color)
#             # print('node: '+str(node)+' child: '+str(child)+' k='+str(i))
#             coloring(gr,child,root,i,d+1)
#             k=k-i
#     if(node==root):
#         #gr.nodes[node]['color']='green'
#         b=[]
#         for n in gr.nodes():
#             if gr.nodes[n]['color']=='blue':
#                 b.append(n)
#         print("Blue nodes: "+str(b))
#     return gr

def NewColoring(g,node,root,k,X):
    if node==root:
        for n in g.nodes:
                g.nodes[n]['color']='red'
                
    children=g.nodes[node]['children']
    # d=nx.shortest_path_length(g,root,node)
    if children == [] and k>0:
        g.nodes[node]['color']='blue'
        return
    if k>0:
         # print('this node:',node)
         if g.nodes[node]['m'+str(len(children)-1)]['k'+str(k)]['bn']['blue']:
             g.nodes[node]['color']='blue'
             # d=0
             # k=k-1
         for c in children[::-1]:
             if c==children[0]:
                if g.nodes[node]['color']=='blue':
                    # print('this node:',node,'l:',d,'k:',k,'color:',g.nodes[node]['color'],'child:',c,'j:',k-1)
                    NewColoring(g,c,root,k-1,X)       
                else:
                    # print('this node:',node,'l:',d,'k:',k,'color:',g.nodes[node]['color'],'child:',c,'j:',k)
                    NewColoring(g,c,root,k,X) 
             else:
                    PreviosChild=g.nodes[node]['m'+str((children.index(c)-1))]
                    Child=g.nodes[c]['minSend']
                 # if g.nodes[node]['color'] == 'blue':
                    j=mSplit(k,PreviosChild,Child,X,g.nodes[node]['color'])
                    NewColoring(g,c,root,j,X)
                    k=k-j
                 
                 # else:
                 #     j=mSplit(g,d,k,PreviosYm,Xm, g.nodes[node]['color'])
                 #     NewColoring(g,c,root,d+1,j)
                 #     k=k-j
                 # print('this node:',node,'l:',d,'k:',k,'color:',g.nodes[node]['color'],'child:',c,'j:',j)
                 # print('PreviosYm',PreviosYm,'xm',Xm)
                
                 
    if(node==root):
        #gr.nodes[node]['color']='green'
        b=[]
        for n in g.nodes():
            if g.nodes[n]['color']=='blue':
                b.append(n)
        print("Blue nodes: "+str(b))
        return b

def messageCount(g,root):
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
        print('i: '+str(i))
        for node in r[depth-i]: 
            # if g.nodes[node]['type'] == 's':
            #     # print(node)
            if node == root:
                return
            perent=list(g.in_edges(nbunch=node))[0][0]
            if g.nodes[node]['color']=='blue':
                att={(perent,node):{'mesageCount':1}}
            else:
                children=list(g.out_edges(nbunch=node))
                if children:
                    s=0
                    for c in children:
                       s=s+ g.edges[c]['mesageCount']
                    att={(perent,node):{'mesageCount':s}}
                else:
                    att={(perent,node):{'mesageCount':g.nodes[node]['load']}}
            nx.set_edge_attributes(g,att)
                
    
