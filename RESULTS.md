# Results: Tawseel failed-delivery baseline

## Baseline table

Last 20% of parcels by time, make_parcels(4000, seed=0), n = 800.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---|---|---|---|
| Dummy (prior) | 0.500 | 0.220 | 0.1717 | 0.22 |
| My model (logistic regression + history features) | 0.628 | 0.323 | 0.1669 | 0.22 |

PR-AUC lift over the dummy: +0.103. On seeds 1, 2 and 3 the lift was 0.107, 0.136 and 0.173, ROC AUC 0.625 to 0.687, and the Brier score was better than the dummy every time.

## Three sentences for Sherif

On parcels created after the training period, the model finds failing deliveries clearly better than guessing: PR-AUC 0.32 against 0.22 for a model that only knows the overall failure rate. It is a modest model (ROC AUC 0.63), so use it to rank parcels by risk, for example to call the customer first, and not to decide which parcels to cancel. It only uses what is known when the parcel is created and was tested on later parcels only, so these numbers should be close to what you will see in production.
