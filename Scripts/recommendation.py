import pandas as pd # type: ignore
from collections import defaultdict # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import mean_squared_error, r2_score # type: ignore
from sklearn.preprocessing import LabelEncoder # type: ignore
import matplotlib.pyplot as plt # type: ignore

df_raw = pd.read_csv('./Datasets/feature_engineered_dataset.csv')

major_business_hubs = [
    'Atlanta, GA', 'Austin, TX', 'Baltimore, MD', 'Boston, MA', 'Charlotte, NC',
    'Chicago, IL', 'Dallas, TX', 'Dallas/Fort Worth, TX', 'Denver, CO', 'Detroit, MI',
    'Houston, TX', 'Indianapolis, IN', 'Las Vegas, NV', 'Los Angeles, CA', 'Miami, FL',
    'Minneapolis, MN', 'Nashville, TN', 'New Orleans, LA', 'New York, NY', 'Newark, NJ',
    'Orlando, FL', 'Philadelphia, PA', 'Phoenix, AZ', 'Portland, OR', 'Raleigh/Durham, NC',
    'Salt Lake City, UT', 'San Antonio, TX', 'San Diego, CA', 'San Francisco, CA',
    'San Jose, CA', 'Seattle, WA', 'St. Louis, MO', 'Tampa, FL', 'Washington, DC'
]

df_hubs = df_raw[
    df_raw['ORIGIN_CITY'].isin(major_business_hubs) &
    df_raw['DEST_CITY'].isin(major_business_hubs)
].copy()

df = df_raw.copy()
drop_cols = ['Unnamed: 0', 'FL_DATE', 'AIRLINE_DOT', 'AIRLINE_CODE', 'DOT_CODE',
             'ORIGIN_CITY', 'DEST_CITY', 'DEP_TIME', 'ARR_TIME', 'CANCELLED',
             'DIVERTED', 'FL_NUMBER', 'TOTAL_KNOWN_DELAY_CAUSE', 'DEP_DELAY']
df = df.drop(columns=drop_cols)

categorical_cols = ['AIRLINE', 'ORIGIN', 'DEST', 'DEP_TIME_OF_DAY', 'ROUTE', 'HAUL_TYPE']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop(columns=['ARR_DELAY'])
y = df['ARR_DELAY']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1, verbose=1)
print("Training Random Forest...")
rf.fit(X_train, y_train)
print("Training complete.")

df_pred_input = df_hubs.drop(columns=drop_cols + ['ARR_DELAY'])
for col in categorical_cols:
    le = label_encoders[col]
    df_pred_input[col] = le.transform(df_pred_input[col])

df_hubs['PREDICTED_ARR_DELAY'] = rf.predict(df_pred_input)

predicted_route_delay = df_hubs.groupby(['ORIGIN_CITY', 'DEST_CITY'])['PREDICTED_ARR_DELAY'].mean().reset_index()
delay_graph = defaultdict(dict)
for _, row in predicted_route_delay.iterrows():
    origin, dest, delay = row['ORIGIN_CITY'], row['DEST_CITY'], row['PREDICTED_ARR_DELAY']
    delay_graph[origin][dest] = delay

def recommend_two_leg_path(graph, origin, destination):
    if destination not in graph.get(origin, {}):
        return "No direct route found."

    direct_delay = graph[origin][destination]
    best_path = None
    best_delay = float('inf')

    for mid in graph[origin]:
        if mid == destination or destination not in graph.get(mid, {}):
            continue
        combined_delay = graph[origin][mid] + graph[mid][destination]
        if combined_delay <= direct_delay and combined_delay < best_delay:
            best_delay = combined_delay
            best_path = (origin, mid, destination)

    if best_path:
        return {
            'direct_predicted_delay': round(direct_delay, 2),
            'recommended_path': best_path,
            'combined_predicted_delay': round(best_delay, 2)
        }
    else:
        return "No better 2-leg path found."

result = recommend_two_leg_path(delay_graph, 'New York, NY', 'San Francisco, CA')
print(result)