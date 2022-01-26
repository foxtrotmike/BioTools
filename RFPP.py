import numpy as np
def Calculate_RFPP(z,Targetlabels):
  RPP=[]
  index_P=np.argsort(z)
  n=len(Targetlabels)
  sorted_index=index_P[::-1][:n]
  sorted_score=z[index_P[::-1][:n]]
  sorted_Targetlabels=Targetlabels[sorted_index]
  ###
  RPP.append(np.where(sorted_Targetlabels==1))
  RFPP=np.min(RPP)+1
  return RFPP 
