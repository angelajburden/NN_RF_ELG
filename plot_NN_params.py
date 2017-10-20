import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np

types = ['recall', 'precision', 'accuracy', 'F1']
for type in types:
    sver = ['lbfgs', 'sgd','adam']
    av,HL,HLN,TP1,TN1,FP1,FN1,prec1,recall1 = np.loadtxt('params_' + sver[0] + '.txt').T
    av,HL,HLN,TP2,TN2,FP2,FN2,prec2,recall2 = np.loadtxt('params_' + sver[1] + '.txt').T
    av,HL,HLN,TP3,TN3,FP3,FN3,prec3,recall3 = np.loadtxt('params_' + sver[2] + '.txt').T
    acc1 = (TP1 + TN1)/(TP1 + TN1 + FP1 + FN1)
    acc2 = (TP2 + TN2)/(TP2 + TN2 + FP2 + FN2)
    acc3 = (TP3 + TN3)/(TP3 + TN3 + FP3 + FN3) 
    F1score1 = 2*(prec1 * recall1)/(prec1+ recall1)
    F1score2 = 2*(prec2 * recall2)/(prec2+ recall2)
    F1score3 = 2*(prec3 * recall3)/(prec3+ recall3)
    fig= plt.figure(figsize=(6, 8))
    n=16
    nrows=4
    ncols=4
    max_val =0.0
    valy=[3,2,1,0]
    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(wspace=0.0, hspace=0.0) # set the spacing between axes.
        
    for i in np.array(range(4)):
    
        nstart =i*n
        nfin =(i+1)*n
        if (type == 'recall'):
        #get rid of recall =1 as this means the solver 
        #predicted everything as 1. set to zero
            data_prec1 = recall1[nstart:nfin]
            data_prec2 = recall2[nstart:nfin]
            data_prec3 = recall3[nstart:nfin]
            data_prec3[data_prec3==1.]=0.0
            data_prec1[data_prec1==1.]=0.0
            data_prec2[data_prec2==1.]=0.0
        if (type == 'precision'):
            data_prec1 = prec1[nstart:nfin]
            data_prec2 = prec2[nstart:nfin]
            data_prec3 = prec3[nstart:nfin]   
        if (type == 'accuracy'):
            data_prec1 = acc1[nstart:nfin]
            data_prec2 = acc2[nstart:nfin]
            data_prec3 = acc3[nstart:nfin]
        if (type == 'F1'):
            data_prec1 = F1score1[nstart:nfin]
            data_prec2 = F1score2[nstart:nfin]
            data_prec3 = F1score3[nstart:nfin]
        
        #shape into matrix with HL nodes and HL as rows&clos    
        grid_prec1 = data_prec1.reshape((nrows,ncols))    
        grid_prec2 = data_prec2.reshape((nrows,ncols))    
        grid_prec3 = data_prec3.reshape((nrows,ncols))            

        ax1 = plt.subplot(gs1[i*3])
        ax2 = plt.subplot(gs1[i*3 +1])
        ax3 = plt.subplot(gs1[i*3 +2])
    
        [row1,col1] = np.argwhere(grid_prec1 == np.max(grid_prec1))[0][0:2]
        #[0][0:2] chooses 1st val if 2 vals the same
        ym1 = valy[row1] 
        xm1=col1  
        [row2,col2] = np.argwhere(grid_prec2 == np.max(grid_prec2))[0][0:2]
        ym2 = valy[row2] 
        xm2=col2
        [row3,col3] = np.argwhere(grid_prec3 == np.max(grid_prec3))[0][0:2]
        ym3 = valy[row3] 
        xm3=col3
    
        if (np.max(grid_prec1)>max_val):
            max_val =np.max(grid_prec1) 
            [[rowm,colm]] = np.argwhere(grid_prec1 == np.max(grid_prec1))
            pmax = (i*3+1)
        if (np.max(grid_prec2)>max_val):
            max_val =np.max(grid_prec2)
            [[rowm,colm]] = np.argwhere(grid_prec2 == np.max(grid_prec2))
            pmax = (i*3+2)
        if (np.max(grid_prec3)>max_val):
            max_val =np.max(grid_prec3)
            [[rowm,colm]] = np.argwhere(grid_prec3 == np.max(grid_prec3))
            pmax = (i*3+3)
        im1= ax1.imshow(grid_prec1,interpolation='nearest', cmap=cm.YlGnBu,vmin=0.0, vmax=1.0, extent =[0,4,0,4]) 
        im2= ax2.imshow(grid_prec2,interpolation='nearest', cmap=cm.YlGnBu,vmin=0.0, vmax=1.0, extent =[0,4,0,4])
        im3= ax3.imshow(grid_prec3,interpolation='nearest', cmap=cm.YlGnBu,vmin=0.0, vmax=1.0, extent =[0,4,0,4])  
        ax1.plot(xm1+0.5,ym1+0.5,'k*')
        ax2.plot(xm2+0.5,ym2+0.5,'k*')
        ax3.plot(xm3+0.5,ym3+0.5,'k*')
        plt.setp(ax1, yticks=[0.5,1.5, 2.5, 3.5], yticklabels=['4', '3', '2', '1'], ylabel='no. HL')
        if (i==3):
            plt.setp(ax1, xticks=[0.5, 1.5, 2.5, 3.5], xticklabels=['10', '20', '50', '100'], xlabel='no. HL nodes')    
            plt.setp(ax2, xticks=[0.5, 1.5, 2.5, 3.5], xticklabels=['10', '20', '50', '100'], xlabel='no. HL nodes')
            plt.setp(ax3, xticks=[0.5, 1.5, 2.5, 3.5], xticklabels=['10', '20', '50', '100'], xlabel='no. HL nodes')
            ax2.set_yticks([])
            ax3.set_yticks([])
        else:
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax3.set_xticks([])
            ax3.set_yticks([]) 

    ymax = valy[rowm] 
    xmax = colm    
    label=max_val
    plt.subplot(gs1[pmax-1])
    plt.plot(xmax+0.5,ymax+0.5,'o',markersize=15, color='red', mfc='none')
    plt.annotate(label, xy=(xmax+0.5, ymax+0.5), xytext=(xmax-20, ymax+20),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8),
            arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        
    plt.subplot(gs1[0]) 
    plt.annotate('alpha',color='blue', xy=(10, 540), xycoords='figure pixels')  
    plt.annotate('solver',color='red', xy=(205, 540), xycoords='figure pixels')
    plt.annotate('1e-1', color='blue', xy=(10, 505), xycoords='figure pixels') 
    plt.annotate('1e-2', color='blue', xy=(10, 392), xycoords='figure pixels') 
    plt.annotate('1e-3', color='blue', xy=(10, 275), xycoords='figure pixels') 
    plt.annotate('1e-4', color='blue', xy=(10, 160), xycoords='figure pixels') 
    plt.annotate(sver[0], color='red', xy=(100, 520), xycoords='figure pixels') 
    plt.annotate(sver[1], color='red', xy=(210, 520), xycoords='figure pixels') 
    plt.annotate(sver[2], color='red', xy=(320, 520), xycoords='figure pixels') 
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    fig.colorbar(im3, cax=cax)
    pad = 5 
    plt.suptitle(type, fontsize=15)                                                                                                                            
    fig.savefig('%s_all.pdf' % type)

