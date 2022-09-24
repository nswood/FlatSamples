import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib 
from sklearn.metrics import auc
inlay_font = {
        'family' : 'normal',
        'weight' : 'normal',
        'size'   : 10
}
axis_font = {
        'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16
}
legend_font = {
        #'family' : 'normal',
        #'weight' : 'normal',
        'fontsize'   : 12
}

_singleton_labels=["zpr_fj_msd","zpr_fj_pt","zpr_fj_eta","zpr_fj_phi","zpr_fj_n2b1","zpr_fj_tau21","zpr_fj_particleNetMD_QCD", "zpr_fj_particleNetMD_Xbb", "zpr_fj_particleNetMD_Xcc", "zpr_fj_particleNetMD_Xqq","zpr_fj_nBHadrons","zpr_fj_nCHadrons", "zpr_genAK8Jet_mass","zpr_genAK8Jet_pt","zpr_genAK8Jet_eta","zpr_genAK8Jet_phi", "zpr_genAK8Jet_partonFlavour","zpr_genAK8Jet_hadronFlavour", "zpr_fj_nBtags","zpr_fj_nCtags","zpr_fj_nLtags","zpr_fj_nparts"]

nbins = 20
_titles={
  "zpr_fj_msd" : {"name" : "Jet $\mathrm{m_{SD}}$ (GeV)", "bins" : np.linspace(30,350,nbins)},
  "zpr_fj_pt"  : {"name" : "Jet $\mathrm{p_T}$ (GeV)",    "bins" : np.linspace(200,1500,nbins)},
  "zpr_fj_eta" : {"name" : "Jet $\mathrm{\eta}$",         "bins" : np.linspace(-3,3,nbins)},
  "zpr_fj_phi" : {"name" : "Jet $\mathrm{\phi}$",         "bins" : np.linspace(-np.pi,np.pi,nbins)},
  "zpr_fj_n2b1" : {"name" : "Jet $\mathrm{N_2}$",         "bins" : np.linspace(0,0.5,nbins)},
  "zpr_fj_tau21" : {"name" : "Jet $\mathrm{\tau_{2,1}}$", "bins" : np.linspace(0,1.0,nbins)},
  "zpr_genAK8Jet_eta" : {"name" : "Generator jet $\mathrm{\eta}$",  "bins" : np.linspace(-3,3,nbins)},
  "zpr_genAK8Jet_phi" : {"name" : "Generator jet $\mathrm{\phi}$",  "bins" : np.linspace(-np.pi,np.pi,nbins)},
  "zpr_genAK8Jet_partonFlavour" : {"name" : "Generator jet parton flavor",  "bins" : nbins},
  "zpr_genAK8Jet_hadronFlavour" : {"name" : "Generator jet hadron flavor",  "bins" : nbins},
  "zpr_genAK8Jet_mass" : {"name" : "Generator jet mass (GeV)",  "bins" : np.linspace(30,250,nbins)},
  "zpr_genAK8Jet_pt"   : {"name" : "Generator jet $\mathrm{p_T}$ (GeV)", "bins" : np.linspace(200,1500,nbins)},
  "zpr_fj_nparts"      : {"name" : "Number of particles", "bins" : np.linspace(0,150,51)},
  "zpr_fj_nBHadrons"   : {"name" : "Number of B hadrons", "bins" : np.linspace(0,10,11)},
  "zpr_fj_nCHadrons"   : {"name" : "Number of C hadrons", "bins" : np.linspace(0,10,11)},

  "zpr_PF_ptrel" :  {"name" : "Particle relative $\mathrm{p_T}$",  "bins" : np.linspace(0,1,nbins)},
  "zpr_PF_etarel" : {"name" : "Particle relative $\mathrm{\eta}$", "bins" : np.linspace(-1,1,nbins)},
  "zpr_PF_phirel" : {"name" : "Particle relative $\mathrm{\phi}$", "bins" : np.linspace(-1,1,nbins)},
  "zpr_PF_dz" :     {"name" : "Particle dz", "bins" : np.linspace(-100,100,nbins)},
  "zpr_PF_d0" :     {"name" : "Particle d0", "bins" : np.linspace(-100,100,nbins)},
  "zpr_PF_pdgId" :  {"name" : "Particle pdgid", "bins" : nbins},

  "zpr_SV_mass"     :  {"name" : "SV mass (GeV)", "bins" : np.linspace(0,180,nbins)},
  "zpr_SV_dlen"     :  {"name" : "SV decay length (cm)" , "bins" : np.linspace(0,250,nbins)},
  "zpr_SV_dlenSig"  :  {"name" : "SV decay length significance" , "bins" : np.linspace(0,6e3,nbins)},
  "zpr_SV_dxy"      :  {"name" : "SV 2D decay length (cm)" , "bins" : np.linspace(0,100,nbins)},
  "zpr_SV_dxySig"   :  {"name" : "SV 2D decay length significance" , "bins" : np.linspace(0,6e3,nbins)},
  "zpr_SV_chi2"     :  {"name" : "SV chi squared/ndof" , "bins" : np.linspace(-5e4,5e4,nbins)},
  "zpr_SV_ptrel"    :  {"name" : "SV relative $\mathrm{p_T}$" , "bins" : np.linspace(0,1,nbins)},
  "zpr_SV_x"        :  {"name" : "SV x position (cm)" , "bins" : np.linspace(-80,80,nbins)},
  "zpr_SV_y"        :  {"name" : "SV y position (cm)" , "bins" : np.linspace(-80,80,nbins)},
  "zpr_SV_z"        :  {"name" : "SV z position (cm)" , "bins" : np.linspace(-150,150,nbins)},
  "zpr_SV_pAngle"   :  {"name" : "SV pointing angle" , "bins" : np.linspace(0,3.5,nbins)},
  "zpr_SV_etarel"   :  {"name" : "SV relative $\mathrm{\eta}$", "bins" : np.linspace(-1,1,nbins)},
  "zpr_SV_phirel"   :  {"name" : "SV relative $\mathrm{\phi}$", "bins" : np.linspace(-1,1,nbins)},


}

_singleton_features_labels=["zpr_fj_jetNSecondaryVertices","zpr_fj_jetNTracks","zpr_fj_tau1_trackEtaRel_0","zpr_fj_tau1_trackEtaRel_1","zpr_fj_tau1_trackEtaRel_2","zpr_fj_tau2_trackEtaRel_0","zpr_fj_tau2_trackEtaRel_1","zpr_fj_tau2_trackEtaRel_3","zpr_fj_tau1_flightDistance2dSig","zpr_fj_tau2_flightDistance2dSig","zpr_fj_tau1_vertexDeltaR","zpr_fj_tau1_vertexEnergyRatio","zpr_fj_tau2_vertexEnergyRatio","zpr_fj_tau1_vertexMass","zpr_fj_tau2_vertexMass","zpr_fj_trackSip2dSigAboveBottom_0","zpr_fj_trackSip2dSigAboveBottom_1","zpr_fj_trackSip2dSigAboveCharm","zpr_fj_trackSip3dSig_0","zpr_fj_trackSip3dSig_0","zpr_fj_tau1_trackSip3dSig_1","zpr_fj_trackSip3dSig_1","zpr_fj_tau2_trackSip3dSig_0","zpr_fj_tau2_trackSip3dSig_1","zpr_fj_trackSip3dSig_2","zpr_fj_trackSip3dSig_3","zpr_fj_z_ratio"]
_p_features_labels=["zpr_PF_ptrel","zpr_PF_etarel","zpr_PF_phirel","zpr_PF_dz","zpr_PF_d0","zpr_PF_pdgId"]
_SV_features_labels=["zpr_SV_mass","zpr_SV_dlen","zpr_SV_dlenSig","zpr_SV_dxy","zpr_SV_dxySig","zpr_SV_chi2","zpr_SV_ptrel","zpr_SV_x","zpr_SV_y","zpr_SV_z","zpr_SV_pAngle","zpr_SV_etarel","zpr_SV_phirel"]

def reshape_inputs(array, n_features):
    array = np.split(array, n_features, axis=-1)
    array = np.concatenate([np.expand_dims(array[i],axis=-1) for i in range(n_features)],axis=-1)
    return array

def train_val_test_split(array,train=0.8,val=0.1,test=0.1):
    n_events = array.shape[0]
    return array[:int(n_events*train)], array[int(n_events*train):int(n_events*(train+val))], array[int(n_events*(train+val)):]

def axis_settings(ax):
    import matplotlib.ticker as plticker
    #ax.xaxis.set_major_locator(plticker.MultipleLocator(base=20))
    ax.xaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(plticker.AutoMinorLocator(5))
    ax.tick_params(direction='in', axis='both', which='major', labelsize=15, length=12)#, labelleft=False )
    ax.tick_params(direction='in', axis='both', which='minor' , length=6)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')    
    #ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
    ax.grid(which='major', alpha=0.9, linestyle='dotted')
    return ax

def makedir(outdir):
    if os.path.isdir(outdir):
        from datetime import datetime
        now = datetime.now()
        outdir += now.strftime("%Y_%D_%H_%M").replace("/","_")
    os.system("mkdir -p "+outdir )

    return 

def plot_features(array, labels, feature_labels, outdir, text_label=None):
    array = np.nan_to_num(array,nan=0.0, posinf=0., neginf=0.) 
    if labels.shape[1]==2: 
        processes = ["Z'","QCD"]
    else: 
        processes = ["Z'(bb)","Z'(cc)","Z'(qq)","QCD"]

    if len(array.shape) == 2:
        for ifeat in range(array.shape[-1]):
            plt.clf()
            fig,ax = plt.subplots() 
            ax = axis_settings(ax)
            for ilabel in range(labels.shape[1]):
               try: 
                   x_label = _titles[feature_labels[ifeat]]["name"]
                   bins    = _titles[feature_labels[ifeat]]["bins"]
               except:
                   x_label = feature_labels[ifeat]
                   bins    = 20

               tmp = array[labels[:,ilabel].astype(bool),ifeat]
               ax.hist(tmp,  
                       label=processes[ilabel], bins=bins, 
                       histtype='step',alpha=0.7, 
                       density=True,
               )
            ax.set_yscale('log')
            ax.set_xlabel(x_label,x=0.7,**axis_font)
            ax.set_ylabel("Normalized counts",y=0.7,**axis_font)
            ax.legend(loc="upper right",**legend_font)
            plt.tight_layout()
            plt.savefig(outdir+'/'+feature_labels[ifeat]+'.png')
            plt.clf()
            if ifeat > 1:
                ibin=0
                for msd_lo,msd_hi in [(40.,80.),(80.,120.),(120.,160.),(160.,250.),(250.,350.)]:
                    fig,ax = plt.subplots()
                    for ilabel in range(labels.shape[1]):
                        msd_idxs = (array[:,0] > msd_lo) & (array[:,0] < msd_hi) & (labels[:,ilabel]==1)
                        tmp = array[msd_idxs,ifeat]
                        ax.hist(tmp,
                                label=processes[ilabel], bins=bins,
                                histtype='step',alpha=0.7,
                                density=True,
                        )
                    ax.text(0.6,0.85,"%.0f < $\mathrm{m_{SD}}$ < %.0f"%(msd_lo,msd_hi),transform=ax.transAxes)
                    ax.set_yscale('log')
                    ax.set_xlabel(x_label,x=0.7,**axis_font)
                    ax.set_ylabel("Normalized counts",y=0.7,**axis_font)
                    ax.legend(loc="upper right",**legend_font)
                    plt.tight_layout()
                    plt.savefig(outdir+'/'+feature_labels[ifeat]+'_msdbin'+str(ibin)+'.png')
                    ibin+=1
    elif len(array.shape) == 3:
        max_parts = 10
        for ifeat in range(array.shape[-1]):
            for ipart in range(array.shape[1]):
                plt.clf()
                fig,ax = plt.subplots() 
                ax = axis_settings(ax)
                for ilabel in range(labels.shape[1]):
                    tmp = array[labels[:,ilabel].astype(bool),ipart,ifeat]
                    try: 
                        x_label = _titles[feature_labels[ifeat]]["name"]
                        bins    = _titles[feature_labels[ifeat]]["bins"]
                    except:
                        x_label = feature_labels[ifeat]
                        bins    = 20
                    ax.hist(tmp,  
                        label=processes[ilabel], bins=bins, 
                        histtype='step',alpha=0.7, 
                        density=True,
                    )
                    ax.text(0.63,0.85,text_label+" "+str(ipart),transform=ax.transAxes,)
                    ax.set_yscale('log')
                    ax.set_xlabel(x_label,x=0.7,**axis_font)
                    ax.set_ylabel("Normalized counts",y=0.7,**axis_font)
                    ax.legend(loc="upper right",**legend_font)
                    plt.tight_layout()
                    plt.savefig(outdir+'/'+'ipart_'+str(ipart)+'_'+feature_labels[ifeat]+'.png')
                if ipart > 10: break
    else:
        raise ValueError("I don't understand this array shape",array.shape)

def plot_loss(loss_vals_training,loss_vals_validation,opath):
    plt.clf()
    fig,ax = plt.subplots()
    ax = axis_settings(ax)
 
    loss_vals_training = loss_vals_training[loss_vals_training!=0]
    loss_vals_validation = loss_vals_validation[loss_vals_validation!=0]

    ax.plot(range(1,len(loss_vals_training)+1), loss_vals_training, lw=2.0,label="training") 
    ax.plot(range(1,len(loss_vals_validation)+1), loss_vals_validation, lw=2.0, label="validation") 
    ax.set_xlabel("Epoch",x=0.7,**axis_font)
    ax.set_ylabel("Loss",y=0.7,**axis_font)
    ax.legend(loc="upper right",**legend_font)
    plt.tight_layout()
    plt.savefig(opath+"/loss.png")

def plot_roc_curve(testLabels, testPredictions, training_text, opath, modelName):
    os.system("mkdir -p "+opath)
    if testLabels.shape[1]==2:
        processes = ["Z'","QCD"]
    else:
        processes = ["Z'(bb)","Z'(cc)","Z'(qq)","QCD"]
    training_text = training_text.split(":")
    for ilabel in range(testLabels.shape[1]):
        plt.clf()
        fig,ax = plt.subplots()
        ax = axis_settings(ax)
        response_l = [] 
        bins=None
        for itruth in range(testLabels.shape[1]):
            response, bins, _ = ax.hist(testPredictions[testLabels[:,itruth]>0,ilabel],
                bins=np.linspace(0.,1.0,100),
                label=processes[itruth],
                histtype='step',alpha=0.7,
                density=True,
                lw=2.0,
            )
            response_l.append(response)
        ax.text(0.63,0.85,"\n".join(training_text),transform=ax.transAxes,**inlay_font)
        ax.set_yscale('log')
        ax.set_xlabel(processes[ilabel] + " output",x=0.7,**axis_font)
        ax.set_xlim(-0.01,1.01)
        ax.set_ylabel("Normalized counts",y=0.7,**axis_font)
        ax.legend(loc="upper right",**legend_font)
        plt.tight_layout()
        plt.savefig(opath+"/%s_response_class_%s.png"%(modelName,ilabel))
  
        tpr = None
        fpr_l = []
        fpr_label_l = []
        for itruth in range(testLabels.shape[1]):
            if itruth == ilabel:
                tpr = [ np.sum(response_l[itruth][ib:])/np.sum(response_l[itruth]) for ib in range(len(bins),0,-1) ]
            else: 
                fpr_l.append([ np.sum(response_l[itruth][ib:])/np.sum(response_l[itruth]) for ib in range(len(bins),0,-1) ])
                fpr_label_l.append(processes[itruth])
        plt.clf()
        fig,ax = plt.subplots()
        ax = axis_settings(ax)
        for i in range(len(fpr_l)):
            fpr = np.round_(fpr_l[i],decimals=4)
            tpr = np.round_(tpr,decimals=4)
            ax.plot(fpr_l[i], tpr, 
                    label = "{process} vs {class_name} (auc={auc:.2f})".format(process=fpr_label_l[i],class_name=processes[ilabel],auc=auc(fpr, tpr)),
                    lw=2.0,
                   )
        ax.set_xlabel("False positive rate",x=0.7,**axis_font)
        ax.set_xlim(0.,1.0)
        ax.set_ylabel("True positive rate",x=0.7,**axis_font)
        ax.set_ylim(0.,1.0)
        ax.axvline(x=0.05,ls='--',lw=1.0,c='magenta')
        ax.text(0.63,0.35,"\n".join(training_text),transform=ax.transAxes,**inlay_font)
        ax.legend(loc="lower right",**legend_font)
        plt.tight_layout()
        plt.savefig(opath+"/%s_roc_class_%s.png"%(modelName,ilabel))

    return 

def sculpting_curves(testQcdPredictions, testQcdKinematics, training_text, opath, modelName, score=""):

    ##This isn't enough bins???
    QcdPredictionsPdf,edges = np.histogram(testQcdPredictions, bins=np.linspace(0.,1.,10000), density=True)

    #tot = 0
    #QcdPredictionsCdf = [] 
    #for i,ih in enumerate(QcdPredictionsPdf):
    #    current = ih*(edges[i+1]-edges[i])
    #    tot += current
    #    QcdPredictionsCdf.append(tot)

    if QcdPredictionsPdf.mean()<0.5:
        QcdPredictionsPdf = 1 - QcdPredictionsPdf
    QcdPredictionsCdf = np.cumsum(QcdPredictionsPdf)*(edges[1]-edges[0])
    pctls = [0.02,0.05,0.10,0.25,0.75,1.00]
    cuts = np.searchsorted(QcdPredictionsCdf,pctls)

    

    sculpting_vars = ["zpr_fj_msd","zpr_fj_pt","zpr_fj_eta","zpr_fj_phi","zpr_genAK8Jet_mass","zpr_genAK8Jet_pt","zpr_genAK8Jet_eta","zpr_genAK8Jet_phi", "zpr_genAK8Jet_partonFlavour","zpr_genAK8Jet_hadronFlavour","zpr_fj_nparts","zpr_fj_nBHadrons","zpr_fj_nCHadrons",]
    training_text = training_text.split(":")
    if score:
        training_text.append(score + "probability")
    for i,label in enumerate(sculpting_vars): 
        plt.clf()
        fig,ax=plt.subplots()
        ax = axis_settings(ax)
        for c,p in zip(cuts,pctls):
            #print("QCD < %.2f"%edges[c], testQcdPredictions[testQcdPredictions<edges[c]])
            KinematicsPassingCut = testQcdKinematics[testQcdPredictions<edges[c],_singleton_labels.index(label)]
            ax.hist(KinematicsPassingCut, 
                    label="$\mathrm{\epsilon_{QCD}}=$%.2f"%(p), 
                    bins=_titles[label]["bins"],
                    histtype='step',
                    alpha=0.7,
                    density=True,
                    lw=2.0,
                    )
        ax.set_xlabel(_titles[label]["name"],x=0.7,**axis_font)
        ax.set_ylabel("Normalized counts",y=0.7,**axis_font)
        ax.text(0.63,0.85,"\n".join(training_text),transform=ax.transAxes,**inlay_font)
        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(opath+"/%s_sculpting_qcd_var%i%s.png"%(modelName,i,"_"+score if score else ""))
        ax.set_yscale('log')
        plt.savefig(opath+"/%s_sculpting_qcd_var%i_log%s.png"%(modelName,i,"_"+score if score else ""))
