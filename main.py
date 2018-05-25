from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib.parse
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import io
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import os

normal_file_raw = 'normalTrafficTraining.txt'
anomaly_file_raw = 'anomalousTrafficTest.txt'

normal_file_parse = 'normalRequestTraining.txt'
anomaly_file_parse = 'anomalousRequestTest.txt'

def parse_file(file_in, file_out):
    fin = open(file_in)
    fout = io.open(file_out, "w", encoding="utf-8")
    lines = fin.readlines()
    res = []
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("GET"):
            res.append("GET" + line.split(" ")[1])
        elif line.startswith("POST") or line.startswith("PUT"):
            url = line.split(' ')[0] + line.split(' ')[1]
            j = 1
            while True:
                if lines[i + j].startswith("Content-Length"):
                    break
                j += 1
            j += 1
            data = lines[i + j + 1].strip()
            url += '?' + data
            res.append(url)
    for line in res:
        line = urllib.parse.unquote(line).replace('\n','').lower()
        fout.writelines(line + '\n')
    print ("finished parse ",len(res)," requests")
    fout.close()
    fin.close()

def loadData(file):
    with open(file, 'r', encoding="utf8") as f:
        data = f.readlines()
    result = []
    for d in data:
        d = d.strip()
        if (len(d) > 0):
            result.append(d)
    return result

def print_result(X_train, X_test, y_train, y_test, clf, clf_name):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    matrix = confusion_matrix(y_test, y_pred)
    TP, FP = matrix[0]
    FN, TN = matrix[1]
    PPV = (TP * 1.0) / (TP + FP)
    TPR = (TP * 1.0) / (TP + FN)
    TNR = (FP * 1.0) / (TN + FP)
    ACC = (TP + TN) * 1.0 /  (TP + TN + FP + FN)
    F1 = 2.0 * PPV * TPR / (PPV + TPR)
    print ("%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f"%(clf_name,PPV,TPR,TNR,ACC,F1))


if not os.path.exists('anomalousRequestTest.txt') or not os.path.exists('normalRequestTraining.txt'):
    parse_file(normal_file_raw,normal_file_parse)
    parse_file(anomaly_file_raw,anomaly_file_parse)


bad_requests = loadData('anomalousRequestTest.txt')
good_requests = loadData('normalRequestTraining.txt')

all_requests = bad_requests + good_requests
yBad = [1] * len(bad_requests)
yGood = [0] * len(good_requests)
y = yBad + yGood

print ("Total requests : ",len(all_requests))
print ("Bad requests: ",len(bad_requests))
print ("Good requests: ",len(good_requests))

vectorizer = TfidfVectorizer(min_df=0.0, analyzer="char", sublinear_tf=True, ngram_range=(3, 3))
X = vectorizer.fit_transform(all_requests)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

print ("Requests for Train: ",len(y_train))
print ("Requests for Test: ",len(y_test))
print ("Use Trigram (n=3). Split Train:Test = 8:2.\n")

lgs = LogisticRegression()
dtc = tree.DecisionTreeClassifier()
linear_svm=LinearSVC(C=1)
rfc = RandomForestClassifier(n_estimators=50)

print ("Machine Learning Algorithms   \tPPV\t\tFPR\t\tTPR\t\tACC\t\tF1")
print_result(X_train, X_test, y_train, y_test,lgs,"Logistic Regression         ")
print_result(X_train, X_test, y_train, y_test,dtc,"Decision Tree               ")
print_result(X_train, X_test, y_train, y_test,linear_svm,"Linear SVM (C=1)             ")
print_result(X_train, X_test, y_train, y_test,rfc,"Random Forest(tree=50)      ")


