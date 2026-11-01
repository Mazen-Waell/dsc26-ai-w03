# AI usage

## Tools I used, and for what

Claude: explained the task page, wrote the first version of tawseel/model.py (pipeline, history features, time split, evaluation) and drafted RESULTS.md. I ran the code and all the numbers in RESULTS.md come from my own runs.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I did not need to change either function: the code passed the 5 public tests on the first run. I then tested the hard cases myself with my own scripts: shuffled row order, shifted index, input not modified, and one row alone against the same row inside a batch. All gave the expected result.

## How I checked that my model has no leak

FEATURE_COLUMNS leaves out courier_notes, delivered_at, failed and parcel_id, the history features only look at strictly earlier parcels, and the test ROC AUC is 0.63 to 0.69 over four seeds, far below the 0.9 leak threshold.
