from sklearn.metrics import f1_score

def precision(Y_true, Y_predict, label):
    print(label)
    print(f"F1-score, clase positiva: {f1_score(Y_true, Y_predict)}")
    print(f"F1-score, clase negativa : {f1_score(Y_true, Y_predict, pos_label=0)}")
