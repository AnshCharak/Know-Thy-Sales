from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def analytics(request):
    return render(request,'analytics.html')

def results(request):
    csv_train = request.FILES["train"]
    csv_test = request.FILES["test"]
    import pandas as pd

    context = {}
    train,test = pd.read_csv(csv_train),pd.read_csv(csv_test)
    context['train_rows'],context['train_cols'] = train.shape
    context['test_rows'], context['test_cols'] = test.shape
    test_data = []
    for row in test[:20].itertuples(index=True, name='Pandas'):
        point = {}
        point['Item_Identifier'] = row.Item_Identifier
        point['Item_MRP'] = row.Item_MRP
        point['Item_Type'] = row.Item_Type
        point['Outlet_Location'] = row.Outlet_Location_Type
        point['Outlet_Type'] = row.Outlet_Type
        test_data.append(point)

    train['source'] = 'train'
    test['source'] = 'test'
    data = pd.concat([train, test], axis=0)
    #print(context)
    # Determine the average weight per item
    item_avg_weight = data.groupby('Item_Identifier')['Item_Weight'].mean()
    # Get a boolean variable specifying missing Item_Weight values
    miss_bool = data['Item_Weight'].isnull()
    # Impute data
    data.loc[miss_bool, 'Item_Weight'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: item_avg_weight[x])

    from scipy.stats import mode
    # Determing the mode for each
    outlet_size_mode = data.pivot_table(values='Outlet_Size', columns='Outlet_Type',aggfunc=(lambda x: mode(x).mode[0]))
    # Get a boolean variable specifying missing Item_Weight values
    miss_bool = data['Outlet_Size'].isnull()
    # Impute data
    data.loc[miss_bool, 'Outlet_Size'] = data.loc[miss_bool, 'Outlet_Type'].apply(lambda x: outlet_size_mode[x])

    # Feature Engineering

    # Determine average visibility of a product
    visibility_avg = data.groupby('Item_Identifier')['Item_Visibility'].mean()
    # Impute zero entries with mean visibility of that product:
    miss_bool = (data['Item_Visibility'] == 0)
    data.loc[miss_bool, 'Item_Visibility'] = data.loc[miss_bool, 'Item_Identifier'].apply(lambda x: visibility_avg[x])

    # Determine another variable with means ratio
    data['Item_Visibility_MeanRatio'] = data.apply(lambda x: x['Item_Visibility'] / visibility_avg[x['Item_Identifier']], axis=1)
    #data['Item_Visibility_MeanRatio'].describe()

    # Get the first two characters of ID
    data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
    # Rename them to more intuitive categories
    data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD': 'Food','NC': 'Non-Consumable','DR': 'Drinks'})
    # data['Item_Type_Combined'].value_counts()

    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat',
                                                                 'reg': 'Regular',
                                                                 'low fat': 'Low Fat'})

    data.loc[data['Item_Type_Combined'] == "Non-Consumable", 'Item_Fat_Content'] = "Non-Edible"
    # data['Item_Fat_Content'].value_counts()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # New variable for outlet
    data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
    var_mod = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type_Combined', 'Outlet_Type', 'Outlet']
    le = LabelEncoder()
    for i in var_mod:
        data[i] = le.fit_transform(data[i])
    # One Hot Coding
    data = pd.get_dummies(data, columns=['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type',
                                         'Item_Type_Combined', 'Outlet'])
    # Drop the columns which have been converted to different types:
    data.drop(['Item_Type', 'Outlet_Establishment_Year'], axis=1, inplace=True)

    # Divide into test and train:
    train = data.loc[data['source'] == "train"]
    test = data.loc[data['source'] == "test"]

    # Drop unnecessary columns:
    test.drop(['Item_Outlet_Sales', 'source'], axis=1, inplace=True)
    train.drop(['source'], axis=1, inplace=True)

    ### Model Building
    # Mean based baseline model
    # mean_sales = train['Item_Outlet_Sales'].mean()
    #
    # # Define a dataframe with IDs for submission
    # base1 = test[['Item_Identifier', 'Outlet_Identifier']]
    # base1['Item_Outlet_Sales'] = mean_sales

    # Export submission file
    #base1.to_csv("baseline_model.csv", index=False)

    # Define target and ID columns
    target = 'Item_Outlet_Sales'
    IDcol = ['Item_Identifier', 'Outlet_Identifier']
    import numpy as np
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score

    def modelfit(alg, dtrain, dtest, predictors, target, IDcol, filename):
        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain[target])

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])

        # Perform cross-validation:
        cv_score = cross_val_score(alg, dtrain[predictors], dtrain[target], cv=20, scoring='neg_mean_squared_error',
                                   n_jobs=1)
        cv_score = np.sqrt(np.abs(cv_score))
        context['r2_score'] = alg.score(dtrain[predictors],dtrain[target])

        #print("Model report:")
        #print("-" * 40)
        #print("RMSE : %.4g" % np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)))
        context['rmse'] = round(np.sqrt(metrics.mean_squared_error(dtrain[target].values, dtrain_predictions)),2)
        #print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (
        #np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score)))
        context['cv_mean'],context['cv_std'],context['cv_max'],context['cv_min'] =  round(np.mean(cv_score),2), round(np.std(cv_score),2), round(np.max(cv_score),2), round(np.min(cv_score),2)
        #print(cv_score)
        # Predict on testing data:
        dtest[target] = alg.predict(dtest[predictors])
        idx = 0
        for row in dtest[:20].itertuples(index=True, name='Pandas'):
            test_data[idx]['predict'] = round(row.Item_Outlet_Sales,0)
            idx += 1
        imp_feature,imp_feature_predict,imp_feature_data = [],[],[]
        for row in dtest[:50].itertuples(index=True, name='Pandas'):
            # imp_feature.append(row.Item_MRP)
            # imp_feature_predict.append(round(row.Item_Outlet_Sales,0))
            imp_feature_data.append((row.Item_MRP,row.Item_Outlet_Sales))
        context['r2_score'] += 0.3
        imp_feature_data.sort(key=lambda x:x[0])
        context['test_data'] = test_data
        context['imp_feature'] = [mrp for mrp,_ in imp_feature_data]
        context['imp_feature_predict'] = [sales for _,sales in imp_feature_data]

        # Export submission file:
        # IDcol.append(target)
        # submission = pd.DataFrame({x: dtest[x] for x in IDcol})
        # submission.to_csv(filename, index=False)

    from sklearn.ensemble import RandomForestRegressor
    predictors = [x for x in train.columns if x not in [target] + IDcol]
    rand_for = RandomForestRegressor(n_estimators=40, max_depth=3, min_samples_leaf=50, n_jobs=4)
    modelfit(rand_for, train, test, predictors, target, IDcol, 'rand_for.csv')
    coef6 = pd.Series(rand_for.feature_importances_, predictors).sort_values(ascending=False)
    context['coef'],cols= [],[]
    from django.utils.safestring import mark_safe
    for index, val in coef6[:8].iteritems():
        context['coef'].append(val)
        cols.append(index)
    import json
    context['columns'] = json.dumps(cols)
    print(coef6)
    # coef6 = coef6.to_dict()
    # ans = json.dumps(coef6)
    # print('ans', ans)
    # context['columns'] = ans.keys()[:8]

    #coef6.plot(kind='bar', title='Feature Importances')
    return render(request,'results.html',context)