"""Serve the restaurant lab locally using the repository's original CSV."""
import argparse
import json
import math
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent / "Dataset.csv"
NAME, CITY, CUISINE = "Restaurant Name", "City", "Cuisines"
RATING, COST = "Aggregate rating", "Average Cost for two"
PRICE, BOOK, DELIVERY = "Price range", "Has Table booking", "Has Online delivery"
FIELDS = [COST, PRICE, "Votes", BOOK, DELIVERY]
COLUMNS = ["Restaurant ID", NAME, CITY, CUISINE, RATING, "Votes", COST, "Currency"]
PAGE_SIZE = 20


def records(frame):
    return json.loads(frame.to_json(orient="records"))


def rating_features(frame):
    return frame[FIELDS].assign(**{field: frame[field].map({"No": 0, "Yes": 1}) for field in (BOOK, DELIVERY)})


def report(model, x, y, split, seed, metrics):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, random_state=seed)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return {"train": len(x_train), "test": len(x_test), **{
        label: round(float(metric(y_test, predictions)), 4) for label, metric in metrics.items()
    }}


@lru_cache(maxsize=1)
def load():
    data = pd.read_csv(CSV)
    rating_data = data.loc[data[RATING] > 0].dropna(subset=FIELDS)
    rating_model = DecisionTreeRegressor(min_samples_leaf=12, random_state=105)
    rating_metrics = report(
        rating_model, rating_features(rating_data), rating_data[RATING], .2, 105,
        {"mae": mean_absolute_error, "r2": r2_score},
    )

    cuisine_data = data.dropna(subset=[NAME, CUISINE, COST])
    names = {name: i for i, name in enumerate(sorted(data[NAME].dropna().unique()))}
    classifier = DecisionTreeClassifier(random_state=10)
    cuisine_metrics = report(
        classifier, cuisine_data[[NAME, COST]].assign(**{NAME: cuisine_data[NAME].map(names)}),
        cuisine_data[CUISINE], .2, 10, {"accuracy": accuracy_score},
    )

    candidates = (
        data.dropna(subset=[NAME, CUISINE, RATING])
        .sort_values([RATING, "Restaurant ID"], ascending=[False, True])
        .drop_duplicates(NAME)
    )
    candidates = candidates.loc[candidates[RATING] >= 4].sort_values(NAME)
    cities = data[CITY].value_counts()
    meta = {
        "rows": len(data), "columns": len(data.columns), "city_count": len(cities),
        "unrated": int((data[RATING] == 0).sum()), "missing_cuisine": int(data[CUISINE].isna().sum()),
        "cities": sorted(cities.index), "top_cities": cities.head(8).to_dict(),
        "candidate_count": len(candidates), "rating": rating_metrics, "cuisine": cuisine_metrics,
        "default_id": int(data.loc[data[NAME] == "Ooma", "Restaurant ID"].iloc[0]),
    }
    return data, rating_model, classifier, names, candidates, meta


def api(route, query):
    data, regressor, classifier, names, candidates, meta = load()

    def param(key, default=""):
        return query.get(key, [default])[0]

    if route == "meta":
        return meta
    if route == "data":
        city, search = param("city"), param("q").strip()
        rows = data if not city else data.loc[data[CITY] == city]
        if search:
            matches = rows[NAME].str.contains(search, case=False, regex=False)
            matches |= rows[CUISINE].str.contains(search, case=False, regex=False, na=False)
            matches |= rows["Restaurant ID"].astype(str).eq(search)
            rows = rows.loc[matches]
        page = max(1, min(int(param("page", "1")), math.ceil(len(rows) / PAGE_SIZE)))
        start = (page - 1) * PAGE_SIZE
        return {"total": len(rows), "page": page, "page_size": PAGE_SIZE,
                "rows": records(rows.iloc[start:start + PAGE_SIZE][COLUMNS])}
    if route == "locations":
        city = param("city") or next(iter(meta["top_cities"]))
        rows = data.loc[data[CITY] == city]
        rated = rows.loc[rows[RATING] > 0, RATING]
        cuisines = rows[CUISINE].dropna().str.split(",").explode().str.strip().value_counts()
        points = rows[["Longitude", "Latitude"]].dropna()
        points = points.loc[(points != 0).any(axis=1)]
        return {"city": city, "count": len(rows), "rated": len(rated),
                "average": round(float(rated.mean()), 2) if len(rated) else None,
                "cuisines": cuisines.head(6).to_dict(),
                "price": rows[PRICE].value_counts().sort_index().to_dict(),
                "coordinates": points.to_numpy().tolist()}
    if route == "restaurant":
        restaurant = data.loc[data["Restaurant ID"] == int(param("id"))]
        if restaurant.empty:
            raise ValueError("Restaurant ID was not found in Dataset.csv.")
        row = restaurant.iloc[0]
        similar = candidates.loc[
            (candidates[CUISINE] == row[CUISINE]) & (candidates[NAME] != row[NAME])
        ].sort_values([RATING, NAME], ascending=[False, True])
        return {"source": records(restaurant[COLUMNS + [PRICE, BOOK, DELIVERY]])[0],
                "recommendations": records(similar.head(5)[COLUMNS])}
    if route == "predict":
        cost, price, votes = float(param("cost")), int(param("price")), int(param("votes"))
        book, delivery = param("book"), param("delivery")
        if (not math.isfinite(cost) or cost < 0 or cost > 1e9 or
                price not in range(1, 5) or not 0 <= votes <= 1e9 or
                book not in {"Yes", "No"} or delivery not in {"Yes", "No"}):
            raise ValueError("Enter valid costs, votes, price tier and service options.")
        features = pd.DataFrame([[cost, price, votes, book, delivery]], columns=FIELDS)
        encoded = rating_features(features)
        path = regressor.decision_path(encoded).indices
        tree = regressor.tree_
        decisions = []
        for node, child in zip(path[:-1], path[1:]):
            index = tree.feature[node]
            field = FIELDS[index]
            threshold = float(tree.threshold[node])
            went_left = tree.children_left[node] == child
            if field in {BOOK, DELIVERY}:
                question = f"{field.removeprefix('Has ').capitalize()} available?"
                answer = "No" if went_left else "Yes"
            else:
                question = f"{field} at or below {threshold:g}?"
                answer = "Yes" if went_left else "No"
            decisions.append({"question": question, "answer": answer,
                              "value": features.iloc[0].to_dict()[field]})
        return {"rating": round(float(regressor.predict(encoded)[0]), 2),
                "decisions": decisions, "leaf_rows": int(tree.n_node_samples[path[-1]])}
    if route == "classify":
        name, cost = param("name"), float(param("cost"))
        if name not in names or not math.isfinite(cost) or not 0 <= cost <= 1e9:
            raise ValueError("Choose a known restaurant and enter a valid cost.")
        features = pd.DataFrame([[names[name], cost]], columns=[NAME, COST])
        return {"cuisine": classifier.predict(features)[0]}
    raise KeyError(route)


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, content_type, status=200, attachment=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if attachment:
            self.send_header("Content-Disposition", 'attachment; filename="Dataset.csv"')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path == "/":
            return self.reply((ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        if url.path == "/Dataset.csv":
            return self.reply(CSV.read_bytes(), "text/csv; charset=utf-8", attachment=True)
        if not url.path.startswith("/api/"):
            return self.reply(b"Not found", "text/plain", 404)
        status = 200
        try:
            payload = api(url.path.removeprefix("/api/"), parse_qs(url.query))
        except (ValueError, OverflowError) as error:
            status, payload = 400, {"error": str(error)}
        except KeyError:
            status, payload = 404, {"error": "Endpoint not found."}
        self.reply(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                   "application/json; charset=utf-8", status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Training models from {CSV.name} with scikit-learn {sklearn_version}…", flush=True)
    load()
    with ThreadingHTTPServer((args.host, args.port), Handler) as server:
        print(f"Restaurant Lab → http://{args.host}:{args.port}", flush=True)
        server.serve_forever()
