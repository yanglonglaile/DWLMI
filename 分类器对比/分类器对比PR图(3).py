
import numpy as np
import csv
import math
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from scipy import interp


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def ReadMyCsv2(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = int(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

def ReadMyCsv3(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:
        counter = 0
        while counter < len(row):
            row[counter] = float(row[counter])      # 转换数据类型
            counter = counter + 1
        SaveList.append(row)
    return

def StorFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return

def GenerateLabel(Sample):
    Label = []
    counter = 0
    while counter < len(Sample) / 2:
        Label.append(1)
        counter = counter + 1
    counter = 0
    while counter < len(Sample) / 2:
        Label.append(0)
        counter = counter + 1
    return Label

def MyConfuse(SampleFeature, SampleLabel):
    # 打乱数据集顺序
    counter = 0
    R = []
    while counter < len(SampleFeature):
        R.append(counter)
        counter = counter + 1
    random.shuffle(R)

    RSampleFeature = []
    RSampleLabel = []
    counter = 0
    while counter < len(SampleFeature):
        RSampleFeature.append(SampleFeature[R[counter]])
        RSampleLabel.append(SampleLabel[R[counter]])
        counter = counter + 1
    print('len(RSampleFeature)', len(RSampleFeature))
    print('len(RSampleLabel)', len(RSampleLabel))
    return RSampleFeature, RSampleLabel

def MyEnlarge(x0, y0, width, height, x1, y1, times, mean_fpr, mean_tpr, thickness=1, color = 'blue'):
    def MyFrame(x0, y0, width, height):
        import matplotlib.pyplot as plt
        import numpy as np

        x1 = np.linspace(x0, x0, num=20)  # 生成列的横坐标，横坐标都是x0，纵坐标变化
        y1 = np.linspace(y0, y0, num=20)
        xk = np.linspace(x0, x0 + width, num=20)
        yk = np.linspace(y0, y0 + height, num=20)

        xkn = []
        ykn = []
        counter = 0
        while counter < 20:
            xkn.append(x1[counter] + width)
            ykn.append(y1[counter] + height)
            counter = counter + 1

        plt.plot(x1, yk, color='k', linestyle=':', lw=1, alpha=1)  # 左
        plt.plot(xk, y1, color='k', linestyle=':', lw=1, alpha=1)  # 下
        plt.plot(xkn, yk, color='k', linestyle=':', lw=1, alpha=1)  # 右
        plt.plot(xk, ykn, color='k', linestyle=':', lw=1, alpha=1)  # 上

        return
    # 画虚线框
    width2 = times * width
    height2 = times * height
    MyFrame(x0, y0, width, height)
    MyFrame(x1, y1, width2, height2)

    # 连接两个虚线框
    xp = np.linspace(x0 + width, x1, num=20)
    yp = np.linspace(y0, y1 + height2, num=20)
    plt.plot(xp, yp, color='k', linestyle=':', lw=1, alpha=1)

    # 小虚框内各点坐标
    XDottedLine = []
    YDottedLine = []
    counter = 0
    while counter < len(mean_fpr):
        if mean_fpr[counter] > x0 and mean_fpr[counter] < (x0 + width) and mean_tpr[counter] > y0 and mean_tpr[counter] < (y0 + height):
            XDottedLine.append(mean_fpr[counter])
            YDottedLine.append(mean_tpr[counter])
        counter = counter + 1

    # 画虚线框内的点
    # 把小虚框内的任一点减去小虚框左下角点生成相对坐标，再乘以倍数（4）加大虚框左下角点
    counter = 0
    while counter < len(XDottedLine):
        XDottedLine[counter] = (XDottedLine[counter] - x0) * times + x1
        YDottedLine[counter] = (YDottedLine[counter] - y0) * times + y1
        counter = counter + 1


    plt.plot(XDottedLine, YDottedLine, color=color, lw=thickness, alpha=1)
    return

def MyConfusionMatrix(y_real,y_predict):
    from sklearn.metrics import confusion_matrix
    CM = confusion_matrix(y_real, y_predict)
    print(CM)
    CM = CM.tolist()
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN:%d, FP:%d, FN:%d, TP:%d' % (TN, FP, FN, TP))
    Acc = (TN + TP) / (TN + TP + FN + FP)
    Sen = TP / (TP + FN)
    Spec = TN / (TN + FP)
    Prec = TP / (TP + FP)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    # 分母可能出现0，需要讨论待续
    print('Acc:', round(Acc, 4))
    print('Sen:', round(Sen, 4))
    print('Spec:', round(Spec, 4))
    print('Prec:', round(Prec, 4))
    print('MCC:', round(MCC, 4))
    Result = []
    Result.append(round(Acc, 4))
    Result.append(round(Sen, 4))
    Result.append(round(Spec, 4))
    Result.append(round(Prec, 4))
    Result.append(round(MCC, 4))
    return Result

def MyAverage(matrix):
    SumAcc = 0
    SumSen = 0
    SumSpec = 0
    SumPrec = 0
    SumMcc = 0
    counter = 0
    while counter < len(matrix):
        SumAcc = SumAcc + matrix[counter][0]
        SumSen = SumSen + matrix[counter][1]
        SumSpec = SumSpec + matrix[counter][2]
        SumPrec = SumPrec + matrix[counter][3]
        SumMcc = SumMcc + matrix[counter][4]
        counter = counter + 1
    print('AverageAcc:',SumAcc / len(matrix))
    print('AverageSen:', SumSen / len(matrix))
    print('AverageSpec:', SumSpec / len(matrix))
    print('AveragePrec:', SumPrec / len(matrix))
    print('AverageMcc:', SumMcc / len(matrix))
    return

def MyRealAndPredictionProb(Real,prediction):
    RealAndPredictionProb = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter][1])
        RealAndPredictionProb.append(pair)
        counter = counter + 1
    return RealAndPredictionProb

def MyRealAndPrediction(Real,prediction):
    RealAndPrediction = []
    counter = 0
    while counter < len(prediction):
        pair = []
        pair.append(Real[counter])
        pair.append(prediction[counter])
        RealAndPrediction.append(pair)
        counter = counter + 1
    return RealAndPrediction

def MyStd(result):
    import numpy as np
    NewMatrix = []
    counter = 0
    while counter < len(result[0]):
        row = []
        NewMatrix.append(row)
        counter = counter + 1
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            NewMatrix[counter1].append(result[counter][counter1])
            counter1 = counter1 + 1
        counter = counter + 1
    StdList = []
    MeanList = []
    counter = 0
    while counter < len(NewMatrix):
        # std
        arr_std = np.std(NewMatrix[counter], ddof=1)
        StdList.append(arr_std)
        # mean
        arr_mean = np.mean(NewMatrix[counter])
        MeanList.append(arr_mean)
        counter = counter + 1
    result.append(MeanList)
    result.append(StdList)
    # 换算成百分比制
    counter = 0
    while counter < len(result):
        counter1 = 0
        while counter1 < len(result[counter]):
            result[counter][counter1] = round(result[counter][counter1] * 100, 2)
            counter1 = counter1 + 1
        counter = counter + 1
    return result

PositiveSample = []     # 对应的AllNodeNum
ReadMyCsv2(PositiveSample, 'PositiveSample.csv')
print('PositiveSample[0]', PositiveSample[0])

NegativeSample = []     # 乱序，对应的AllNodeNum
ReadMyCsv2(NegativeSample, 'NegativeSample.csv')
print('NegativeSample[0]', NegativeSample[0])

NewRandomList = []      # 对应的LDLncDiseaseNum(PositiveSampleNum)
ReadMyCsv2(NewRandomList, 'NewRandomList.csv')
# print('NewRandomList[0]', NewRandomList[0])

#############随机森林
AllRealRM=[]
AllPredictionProbRM=[]
AllRealAB = []
AllPredictionProbAB = []
AllPredictionProbLR = []
AllRealLR = []
AllRealNB = []
AllPredictionProbNB = []
AllRealSVM = []
AllPredictionProbSVM= []

mean_fpr = np.linspace(0, 1, 1000)
i = 0
colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']
RealAndPrediction = []
AllResult = []
counter = 0
while counter < 5:
    AllNodeFeatureNum = []
    AllNodeFeatureNumName = 'AllNodeAttributeMannerNum' + str(counter) + '.csv'
    # print('AllNodeFeatureNumName', AllNodeFeatureNumName)
    ReadMyCsv3(AllNodeFeatureNum, AllNodeFeatureNumName)
    PositiveSampleFeature = []
    counter1 = 0
    while counter1 < len(PositiveSample):
        FeaturePair = []
        FeaturePair.extend(AllNodeFeatureNum[PositiveSample[counter1][0]])
        FeaturePair.extend(AllNodeFeatureNum[PositiveSample[counter1][1]])
        PositiveSampleFeature.append(FeaturePair)
        counter1 = counter1 + 1
    NegativeSampleFeature = []
    counter1 = 0
    while counter1 < len(NegativeSample):
        FeaturePair = []
        FeaturePair.extend(AllNodeFeatureNum[NegativeSample[counter1][0]])
        FeaturePair.extend(AllNodeFeatureNum[NegativeSample[counter1][1]])
        NegativeSampleFeature.append(FeaturePair)
        counter1 = counter1 + 1
    Num = 0
    NewPositiveSampleFeature = []
    NewNegativeSampleFeature = []
    counter2 = 0
    while counter2 < len(NewRandomList):
        PairP = []
        PairN = []  #
        PairP.extend(PositiveSampleFeature[Num:Num + len(NewRandomList[counter2])])
        PairN.extend(NegativeSampleFeature[Num:Num + len(NewRandomList[counter2])])
        NewPositiveSampleFeature.append(PairP)
        NewNegativeSampleFeature.append(PairN)
        Num = Num + len(NewRandomList[counter2])
        counter2 = counter2 + 1

    TrainPositiveFeature = []
    TrainNegativeFeature = []
    TestPositiveFeature = []
    TestNegativeFeature = []

    NumTest = 0
    NumTrain = 0
    counter4 = 0
    while counter4 < len(NewRandomList):           # 5次
        if counter4 == counter:
            TestPositiveFeature.extend(NewPositiveSampleFeature[counter4])
            TestNegativeFeature.extend(NewNegativeSampleFeature[counter4])
            NumTest = NumTest + 1
        if counter4 != counter:
            TrainPositiveFeature.extend(NewPositiveSampleFeature[counter4])
            TrainNegativeFeature.extend(NewNegativeSampleFeature[counter4])
            NumTrain = NumTrain + 1
        counter4 = counter4 + 1
    TrainFeature = []
    TrainFeature = TrainPositiveFeature
    TrainFeature.extend(TrainNegativeFeature)
    TrainLabel = GenerateLabel(TrainFeature)
    TrainFeature, TrainLabel = MyConfuse(TrainFeature, TrainLabel)


    TestFeature = []
    TestFeature = TestPositiveFeature
    TestFeature.extend(TestNegativeFeature)
    TestLabel = GenerateLabel(TestFeature)
    TestFeature, TestLabel = MyConfuse(TestFeature, TestLabel)

    # print('np.array(TrainFeature).shape', np.array(TrainFeature).shape)
    # print('np.array(TestFeature).shape', np.array(TestFeature).shape)
    print('start train')

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
    from sklearn.neighbors import KNeighborsClassifier

    model1 = RandomForestClassifier(n_estimators=99)
    model1.fit(TrainFeature, TrainLabel)

    model2 = BaggingClassifier()
    model2.fit(TrainFeature, TrainLabel)

    model3 = AdaBoostClassifier()
    model3.fit(TrainFeature, TrainLabel)

    model4 = GaussianNB()
    model4.fit(TrainFeature, TrainLabel)

    model5 = DecisionTreeClassifier(max_depth=30)
    model5.fit(TrainFeature, TrainLabel)

    y_scoreRM1 = model1.predict_proba(TestFeature)
    y_scoreAB1 = model2.predict_proba(TestFeature)
    y_scoreLR1 = model3.predict_proba(TestFeature)
    y_scoreNB1 = model4.predict_proba(TestFeature)
    y_scoreSVM1 = model5.predict_proba(TestFeature)

    AllRealRM.append(TestLabel)
    AllPredictionProbRM.append(y_scoreRM1[:, 1])
    AllRealAB.append(TestLabel)
    AllPredictionProbAB.append(y_scoreAB1[:, 1])
    AllRealLR.append(TestLabel)
    AllPredictionProbLR.append(y_scoreLR1[:, 1])
    AllRealNB.append(TestLabel)
    AllPredictionProbNB.append(y_scoreNB1[:, 1])
    AllRealSVM.append(TestLabel)
    AllPredictionProbSVM.append(y_scoreSVM1[:, 1])
    counter = counter + 1


import numpy

# 画五折均值的roc和auc




AllRealRM=numpy.concatenate(AllRealRM)
AllPredictionProbRM=numpy.concatenate(AllPredictionProbRM)
precisionRM, recallRM,thresholds= precision_recall_curve(AllRealRM, AllPredictionProbRM)
meanaupr=auc(recallRM, precisionRM)
precisionRM=precisionRM.tolist()
recallRM=recallRM.tolist()
precisionRM.insert(1, 0)
recallRM.insert(0, 1)
plt.plot(precisionRM, recallRM, color='black',
         label=r'RandomForest(AUPR = %0.4f)' % (0.9141),
         lw=2,alpha=1)


AllRealAB=numpy.concatenate(AllRealAB)
AllPredictionProbAB=numpy.concatenate(AllPredictionProbAB)
print(AllRealAB)
print(AllPredictionProbAB)
precisionAB, recallAB,thresholds= precision_recall_curve(AllRealAB, AllPredictionProbAB)
meanaupr=auc(recallAB, precisionAB)
precisionAB=precisionAB.tolist()
recallAB=recallAB.tolist()
precisionAB.insert(1, 0)
recallAB.insert(0, 1)
plt.plot(precisionAB, recallAB, color='red',
         label=r'Bagging(AUPR = %0.4f)' % (meanaupr),
         lw=2,linestyle=':', alpha=1)



AllRealLR=numpy.concatenate(AllRealLR)
AllPredictionProbLR=numpy.concatenate(AllPredictionProbLR)
precisionLR, recallLR,thresholds= precision_recall_curve(AllRealLR, AllPredictionProbLR)
meanaupr=auc(recallLR, precisionLR)
precisionLR=precisionLR.tolist()
recallLR=recallLR.tolist()
precisionLR.insert(1, 0)
recallLR.insert(0, 1)
plt.plot(precisionLR, recallLR, color='blue',
         label=r'AdaBoost(AUPR = %0.4f)' % (meanaupr),
         lw=2,linestyle=':', alpha=1)


AllRealNB=numpy.concatenate(AllRealNB)
AllPredictionProbNB=numpy.concatenate(AllPredictionProbNB)
precisionNB, recallNB,thresholds= precision_recall_curve(AllRealNB, AllPredictionProbNB)
meanaupr=auc(recallNB, precisionNB)
precisionNB=precisionNB.tolist()
recallNB=recallNB.tolist()
precisionNB.insert(1, 0)
recallNB.insert(0, 1)
plt.plot(precisionNB, recallNB, color='green',
         label=r'Naive Bayes(AUPR = %0.4f)' % (meanaupr),
         lw=2,linestyle=':', alpha=1)


AllRealSVM=numpy.concatenate(AllRealSVM)
AllPredictionProbSVM=numpy.concatenate(AllPredictionProbSVM)
precisionSVM, recallSVM,thresholds= precision_recall_curve(AllRealSVM, AllPredictionProbSVM)
meanaupr=auc(recallSVM, precisionSVM)
precisionSVM=precisionSVM.tolist()
recallSVM=recallSVM.tolist()
precisionSVM.insert(1, 0)
recallSVM.insert(0, 1)
plt.plot(precisionSVM, recallSVM, color='purple',
         label=r'DecisionTree(AUPR = %0.4f)' % (meanaupr),
         lw=2,linestyle=':', alpha=1)


plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall', fontsize=13)
plt.ylabel('Precision', fontsize=13)
# plt.title('Receiver operating characteristic')
plt.legend(loc='lower left')
plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
plt.savefig('分类器AUPR.svg')
plt.show()