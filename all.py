def NN(X_train, X_test, file, colname):
    %matplotlib notebook
    import numpy as np
    import pandas as pd
    import seaborn as sn
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    np.set_printoptions(precision=2)


    ar = pd.read_csv(file)

    feature_univ_critertia_ar = [colname]


    X = ar[feature_univ_critertia_ar]
    y = ar['RANK']
    target_class = ['1', '2', '3','4', '5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

    X_univ_2d = ar[[colname]]
    y_univ_2d = ar['RANK']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    Xr_train = X_train[colname].values.reshape(-1, 1)
    Xr_test = X_test[colname].values.reshape(-1, 1)

    from sklearn.preprocessing import MinMaxScaler
    #from sklearn import preprocessing
    scaler = MinMaxScaler()
    #scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(Xr_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(Xr_test)



    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(X_train_scaled, y_train)

    input_univ = [[1],[1],[1],[1],[1],[1]]

    example_univ = input_univ
    #np.array(example_univ)






    X_train = X_train[colname].values.reshape(-1, 1)
    X_test = X_test[colname].values.reshape(-1, 1)


    from sklearn import preprocessing
    #scaler = MinMaxScaler()
    scaler = preprocessing.MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # we must apply the scaling to the test set that we computed for the training set
    X_test_scaled = scaler.transform(X_test)



    knn = KNeighborsClassifier(n_neighbors = 8)
    knn.fit(X_train_scaled, y_train)
    print('Accuracy of ' + colname +  ' K-NN classifier on training set: {:.2f}'
         .format(knn.score(X_train_scaled, y_train)))
    print('2 Accuracy of  ' + colname +  ' K-NN classifier on test set: {:.2f}'
     .format(knn.score(X_test_scaled, y_test)))
    example_univ= [input_univ[0]]

    #np.array(example_univ)

    # print('5 Predicted Academic Reputation QS Rank is', example_univ, ' is ',
    #       target_class[knn.predict(example_univ)[0]-1])

    return int(target_class[knn.predict(example_univ)[0]-1])
# Academic Reputation Score
# Employer Reputation Score
# Faculty Student Score
# Citations per Faculty Score
# International Faculty Score
# International Students Score
# NN(X_train, X_test, 'ar.csv', 'Academic Reputation Score')
#NN(X_train, X_test, 'epf.csv', 'Employer Reputation Score')
NN(X_train, X_test, 'fs.csv', 'Faculty Student Score')
# NN(X_train, X_test, 'cpf.csv', 'Citations per Faculty Score')
# NN(X_train, X_test, 'ifr.csv', 'International Faculty Score')
# NN(X_train, X_test, 'isr.csv', 'International Students Score')
