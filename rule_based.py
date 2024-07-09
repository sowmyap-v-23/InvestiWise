# Rule_based_approach

def predict_investment_risk(credit_rating_impact, total_E, total_S, total_G, predicted_sentiment):
    # Combine ESG scores
    combined_ESG = total_E + total_S + total_G
    
    if credit_rating_impact == 0:
        a = "Stable Financial health"
    else:
        a = "High Risk Unstable Finacial health"
    
    # Determine news sentiment description
    if predicted_sentiment == 1:
        b = "Positive News"
    elif predicted_sentiment == -1:
        b = "Negative news"
    else:
        b = "Neutral news" 
    
    # Determine ESG score description
    if combined_ESG >= 45:
        c = "positive ESG score"
    elif combined_ESG >= 30:
        c = "Moderate ESG score"
    else:
        c = "Low ESG score"
    
    # Construct investment risk prediction string
    if credit_rating_impact == 0:
        investment_risk = "low to moderate investment risk"
    else:
        investment_risk = "High investment risk considering financial health"
    
    prediction_string = f"{a} with {b} with {c} - {investment_risk}"
    return prediction_string

