def HSI(Temp200):
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA

    Temp200_Eq=np.nanmean(Temp200[:,50:166,:],axis=1)
    Temp200_7510N=np.nanmean(Temp200[:,166:,:],axis=1)
    Temp200_7510S=np.nanmean(Temp200[:,0:50,:],axis=1)

    CCAM_HSIK=np.empty(shape=(1812,386-77))
    CCAM_HSIR=np.empty(shape=(1812,386-77))

    for i in range (77,386):
        CCAM_HSIK[:,i-77] = Temp200_Eq[:,i+77] - Temp200_Eq[:,i-77]
        CCAM_HSIR[:,i-77] = (Temp200_7510N[:,i] + Temp200_7510S[:,i])/2 - Temp200_Eq[:,i]

    HSI_C = [CCAM_HSIK,CCAM_HSIR]
    HSI_C = np.reshape(HSI_C,[2,1812*(386-77)])
    HSI_C = np.transpose(HSI_C)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(HSI_C)
    principalDf = pd.DataFrame(data = principalComponents
                 , columns = ['principal component 1', 'principal component 2'])

    CCAM_HSI1 = np.reshape(principalComponents[:,0],[1812,386-77])
    CCAM_HSI2 = np.reshape(principalComponents[:,1],[1812,386-77])

    return(CCAM_HSIK,CCAM_HSIR,CCAM_HSI1,CCAM_HSI2)
