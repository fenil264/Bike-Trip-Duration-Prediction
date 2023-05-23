from django.shortcuts import render
from django.http import HttpResponse
import random

# Create your views here.

def home(request):
    return render(request, 'policy/home.html')


def claim(request):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv("D:/Django/CGC/policy/templates/policy/Project.csv")
    # print("The first 5 rows of the dataframe")
    # df.head(200)
    #data import

    """***Pre Processing***"""

    # df.dtypes  #datatype
    #
    #
    # df.isnull().sum()

    # print(df['Gender'].value_counts())
    df['Gender'] = df['Gender'].replace([np.nan],'Male')  #replace null values in gender column

    df['Age'].fillna(int(df['Age'].mean()), inplace=True) #replace null values in Age column

    df['Treatment_cost'].fillna(int(df['Treatment_cost'].mean()), inplace=True) #replace null values in Treatment_cost column

    df['Insurance_limit'].fillna(int(df['Insurance_limit'].mean()), inplace=True) #replace null values in Insurance Limit column

    # df.head(200)
    #
    # df.isnull().sum()

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label = le.fit_transform(df['Gender'])
    #Labeling our Gender Column

    df.drop("Gender", axis=1, inplace=True)
    df["Gender"] = label
    # df

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    label = le.fit_transform(df['Disease'])


    df.drop("Disease", axis=1, inplace=True)
    df["Disease"] = label
    # df

    # corr = df.corr()
    # plt.figure(figsize=(10,4))
    # sns.heatmap(corr,annot=True,linewidths=0.5,cmap='twilight')
    # plt.show()

    """**`*`Spliting Dataset in Training and testing part`*`**"""

    from re import X
    from sklearn.model_selection import train_test_split

    #divide x_data and y_data
    y_data = df['Claim']
    x_data=df.drop(['Claim','ID'],axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=1)
    #random_state=1 gives you different results everytime

    # print("number of test samples :", x_test.shape[0])
    # print("number of training samples:",x_train.shape[0])
    # print(x_train)


    ## Decision tree
    from sklearn.tree import DecisionTreeClassifier
    decisionTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    decisionTree.fit(x_train,y_train)

    preTree = decisionTree.predict(x_test)

    if request.method == 'POST':
        if request.POST.get('gender')=="Male":
            gender = 1
        elif request.POST.get('gender')=="Female":
            gender = 0

        if request.POST.get('disease')=="Heart Attack":
            disease = 4
        elif request.POST.get('disease')=="Covid-19":
            disease = 1
        elif request.POST.get('disease')=="Injury":
            disease = 5
        elif request.POST.get('disease')=="Cancer":
            disease = 0
        elif request.POST.get('disease')=="Flu":
            disease = 2
        elif request.POST.get('disease')=="HIV":
            disease = 3

        treatment=request.POST.get('treatment_cost')
        insurance=request.POST.get('insurance_limit')
        theclaim = decisionTree.predict([[request.POST.get('hospi_id'),request.POST.get('age'),request.POST.get('min_hospitalized_days'),treatment,insurance,gender,disease]])
        print(theclaim)

        perc = int(treatment)-int(insurance)
        x1 = int(treatment)-perc
        x2 = x1*100/int(treatment)
        if theclaim == [1]:
            if int(treatment) > int(insurance):
                msg = "Your Insurance Claim is "+str(x2)+ "% Approved"
            elif int(treatment) < int(insurance):
                msg = "Your Insurance Claim is 100% Approved"
        elif theclaim == [0]:
            msg = "Your Insurance Claim is Rejected"




    #Confusion Matrix
    # import seaborn as sns
        from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score
    # cm=confusion_matrix(y_test, preTree)
    # ax = sns.heatmap(cm, annot=True, fmt='g')
    # ax.xaxis.set_ticklabels(['0', '1'])
    # ax.yaxis.set_ticklabels(['0', '1'])
    # plt.show()

        print('Precision: %.3f' % precision_score(y_test, preTree, average='macro'))
        print('Recall: %.3f' % recall_score(y_test, preTree, average='macro'))
        print('Accuracy: %.3f' % accuracy_score(y_test, preTree))
        print('F1 Score: %.3f' % f1_score(y_test, preTree, average='macro'))


    else:
        theclaim = 'ERROR: Select an option to generate claim'


    return render(request, 'policy/claim.html', {'claim': msg})
