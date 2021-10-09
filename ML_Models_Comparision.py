

mport numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
# for feature selection
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from  sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.linear_model as lm
# decision tree regresssion feature selectin method
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
#decision tree classification decision tree
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
#random forest feature importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
#matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
# get traininig time
import time

df=pd.read_csv('182_10min_features_combine_spring.csv',dtype={'Time of the day':float,'Weekday_weekend':float,
                                                              'Day_period':float,
                                                              'Outdoor_temperature':float,'Outdoor_humidity':float,
                                                              'Solar_irradiance':float, 'Outdoor_velocity':float,
                                                              'Outdoor_illumination':float,'Rain/no_rain':float,
                                                              'Indoor_CO2':float,'Indoor_temperature':float,'Indoor_humidity':float,
                                                              'Thermal_setpoint_temperature':float, 'Indoor_luminosity':float,
                                                                     'Window_blind':float,'Window_shade':float,'Window_autolock_status':float,
                                                                     'Light_load':float,'Plug_load':float,
                                                               'Occupancy_ratio':float})

df_notime = df.drop(columns=['Time'], axis=1)

"""split training and testing dataset"""
# =============================================================================
# X = df_notime.iloc[:, :-1].values
# y = df_notime.iloc[:, -1].values
# =============================================================================

X = df_notime.drop(columns=['Occupancy_ratio'], axis=1)
y = df_notime['Occupancy_ratio']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# =============================================================================
# 
# df_training=pd.concat([X_train,y_train],axis=1)
# =============================================================================
"""standardisation"""
sc= StandardScaler()
X_train[['Time of the day','Weekday_weekend','Day_period','Outdoor_temperature','Outdoor_humidity',
         'Solar_irradiance','Outdoor_velocity','Outdoor_illumination','Rain/no_rain','Indoor_CO2',
         'Indoor_temperature','Indoor_humidity', 'Thermal_setpoint_temperature', 
        'Indoor_luminosity','Window_blind','Window_autolock_status','Light_load','Plug_load']] = sc.fit_transform(X_train[['Time of the day','Weekday_weekend','Day_period','Outdoor_temperature','Outdoor_humidity',
         'Solar_irradiance','Outdoor_velocity','Outdoor_illumination','Rain/no_rain','Indoor_CO2',
         'Indoor_temperature','Indoor_humidity', 'Thermal_setpoint_temperature', 
        'Indoor_luminosity','Window_blind','Window_autolock_status','Light_load','Plug_load']])

X_test[['Time of the day','Weekday_weekend','Day_period','Outdoor_temperature','Outdoor_humidity',
         'Solar_irradiance','Outdoor_velocity','Outdoor_illumination','Rain/no_rain','Indoor_CO2',
         'Indoor_temperature','Indoor_humidity', 'Thermal_setpoint_temperature', 
        'Indoor_luminosity','Window_blind','Window_autolock_status','Light_load','Plug_load']] = sc.transform(X_test[['Time of the day','Weekday_weekend','Day_period','Outdoor_temperature','Outdoor_humidity',
         'Solar_irradiance','Outdoor_velocity','Outdoor_illumination','Rain/no_rain','Indoor_CO2',
         'Indoor_temperature','Indoor_humidity', 'Thermal_setpoint_temperature', 
        'Indoor_luminosity','Window_blind','Window_autolock_status','Light_load','Plug_load']])
                    
                    
"""1. Logistic Regression"""
lr = LogisticRegression(solver='liblinear',penalty='l1', C=10)

lr.fit(X_train, y_train)

validation_score = cross_validate(lr, X_train, y_train, scoring = 'f1_weighted', cv= 10, return_train_score= True)
validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
#########################################################################
start = time.time()
lr = LogisticRegression(solver='liblinear',penalty='l1', C=10)

lr.fit(X_train, y_train)

# build up prediction model
y_predict = lr.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")


"""2. Support vector machine"""
svm = SVC(kernel="linear")

svm.fit(X_train, y_train)

validation_score = cross_validate(svm, X_train, y_train, scoring = 'f1_weighted', cv= 10, return_train_score= True)

validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
####################################################
start = time.time()
svm = SVC(kernel="linear")

svm.fit(X_train, y_train)
# build up prediction model
y_predict = svm.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")

"""3. Decision tree"""
dt = tree.DecisionTreeClassifier(max_depth = 9, criterion='gini', max_features =3,  #with 18 features 
                            min_samples_leaf = 10, min_samples_split = 7)

# =============================================================================
# dt = tree.DecisionTreeClassifier(max_depth = 10, criterion='gini', max_features= 4,
#                             min_samples_leaf = 10, min_samples_split = 3)
# =============================================================================
dt.fit(X_train, y_train)

validation_score = cross_validate(dt, X_train, y_train, scoring = 'f1_weighted', cv= 10, return_train_score= True)

validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
##########################################################
start = time.time()

# =============================================================================
# dt = tree.DecisionTreeClassifier(criterion='gini',max_depth = 10, min_samples_leaf = 10,
#                                  max_features = 4, min_samples_split = 3)
# =============================================================================

dt = tree.DecisionTreeClassifier(max_depth = 9, criterion='gini', max_features =3,  #with 18 features 
                            min_samples_leaf = 10, min_samples_split = 7)

dt.fit(X_train, y_train)

# build up prediction model
y_predict = dt.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")

"""Gradient boost decision tree"""
gbdt= GradientBoostingClassifier(n_estimators=300, learning_rate=1, max_depth=1, random_state=0)

# =============================================================================
# gbdt= GradientBoostingClassifier(n_estimators=300, learning_rate=0.2, max_depth=1, random_state=0)
# =============================================================================

gbdt.fit(X_train,y_train)

validation_score = cross_validate(gbdt, X_train, y_train, scoring = 'f1_weighted', cv= 10, return_train_score= True)

validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
######################################################
start = time.time()

gbdt= GradientBoostingClassifier(n_estimators=300, learning_rate=1, max_depth=1, random_state=0)

gbdt.fit(X_train,y_train)

# build up prediction model
y_predict = gbdt.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")

"""Random forest"""
cv = StratifiedKFold(10)

rf = RandomForestClassifier(n_estimators=1000, max_depth = 10, bootstrap=True, criterion='gini', max_features =3,
                            min_samples_leaf = 12, min_samples_split = 8 )


rf.fit(X_train, y_train)

validation_score = cross_validate(rf, X_train, y_train, scoring = 'f1_weighted', cv= cv, return_train_score= True)

validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
###############################################################
start = time.time()

# =============================================================================
# rf = RandomForestClassifier(n_estimators=1500, max_depth = 15, bootstrap=True, criterion='gini', max_features =3,
#                             min_samples_leaf = 12, min_samples_split = 8 ) # without 
# =============================================================================


rf = RandomForestClassifier(n_estimators=1000, max_depth = 10, bootstrap=True, criterion='gini', max_features =3,
                            min_samples_leaf = 12, min_samples_split = 8 ) # with using feature selection

rf.fit(X_train, y_train)

# build up prediction model
y_predict = rf.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")

"""Artificial neural network"""
# gridsearchcv
cv = StratifiedKFold(10)
ann = MLPClassifier(hidden_layer_sizes=(18,),max_iter=2000)

parameter_space = { 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'],
                   'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive'],}

grid_search = GridSearchCV(ann, parameter_space,n_jobs=-1, cv=cv)

grid_result = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# determine overfitting or not
print('Mean train score: {}'.format(grid_search.cv_results_['mean_train_score']))
print('Mean test score: {}'.format(grid_search.cv_results_['mean_test_score']))

#################################################################################
ann = MLPClassifier(hidden_layer_sizes=(60,),max_iter=2000, activation='tanh',solver= 'adam',
                    alpha= 0.0001,learning_rate='adaptive' )

ann.fit(X_train, y_train)

validation_score = cross_validate(ann, X_train, y_train, scoring = 'f1_weighted', cv= 10, return_train_score= True)

validation_score['train_score']
Training_score_mean = np.mean(validation_score['train_score'])
print(Training_score_mean)

validation_score['test_score']
validation_score_mean = np.mean(validation_score['test_score'])
print(validation_score_mean)
print(validation_score)
##############################################################
start = time.time()

ann = MLPClassifier(hidden_layer_sizes=(60,),max_iter=1000, activation='tanh',solver= 'adam',
                    alpha= 0.001,learning_rate='adaptive' )

ann.fit(X_train, y_train)

# build up prediction model
y_predict = ann.predict(X_test) 

print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
print(f1_score(y_test,y_predict,average='micro'))
print(roc_auc_score(y_test,y_predict,average=None))

stop = time.time()
print(f"Training time: {stop - start}s")
































































































































