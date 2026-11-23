# Restaurant Lab

The dashboard wraps all four internship tasks and reads the original
[`Dataset.csv`](../Dataset.csv) directly. The two decision trees train when the
server starts. Each prediction is computed on demand using the form values.

## Run locally

From the repository root with Python 3.11 or newer:

```sh
python -m pip install -r ui/requirements.txt
python ui/run.py
```

Open <http://127.0.0.1:8000>. Pass `--port 8001` if port 8000 is in use.

| View | What you can do |
| --- | --- |
| Original data | Search by restaurant, cuisine or ID, filter by city, page through records, select a row and download the source CSV. |
| Live models | Predict a numerical rating or a full cuisine label and compare the result to the selected record. |
| Recommendations | Find up to five restaurants with an identical full cuisine label and recorded ratings of at least 4.0. |
| Locations | Compare city record counts and explore coordinates, common cuisines, price tiers and average ratings. |

The rating model uses a decision tree with a minimum of 12 rows per leaf,
an 80/20 split and a fixed seed. It omits rating color and text, which
encode the target, and trains on rated records. Cuisine classification
keeps the notebook's decision tree and name/cost inputs. A decision tree
does not require the notebook's feature scaling. Its accuracy measures
exact matches of the full cuisine string on a held-out row split, so
restaurants with the same name can appear in both sets. Cost values use
the original recorded units without conversion. Metrics are calculated
using the current CSV at startup.
