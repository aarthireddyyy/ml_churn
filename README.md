ML Churn Prediction 

Problem statement : A subscription business loses revenue every time a customer cancels. Acquiring a new customer costs far more than retaining an existing one. If we can identify customers likely to leave before they do, the retention team can intervene with offers, outreach or support, but only for a limited number of customers because retention budgets are finite.

Success criteria:
Beat the baseline: A dummy model that predicts "no churn" for everyone gets about 73-75% accuracy on the Telco data, and 0% recall. Any real model must clearly beat that on recall and PR-AUC.
Primary metric: Recall on churners at an acceptable precision, plus PR-AUC for threshold-independent comparison.
Interpretability: We should be able to explain which factors drive churn, since a retention team won't act on a black box.
Deployable: The final model must serve predictions through an API in well under a second.