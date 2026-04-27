def classify_risk(prob):
    if prob > 0.7:
        return "High"
    elif prob > 0.4:
        return "Medium"
    else:
        return "Low"
