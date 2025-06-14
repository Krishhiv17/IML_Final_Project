import pandas as pd #type: ignore
from collections import defaultdict #type: ignore
from sklearn.ensemble import RandomForestRegressor #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from sklearn.metrics import mean_squared_error, r2_score #type: ignore
from sklearn.preprocessing import LabelEncoder #type: ignore
import matplotlib.pyplot as plt #type: ignore
import networkx as nx #type: ignore

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

df_all = df_raw.copy()
df_pred_input = df_all.drop(columns=drop_cols + ['ARR_DELAY'])
for col in categorical_cols:
    le = label_encoders[col]
    df_pred_input[col] = le.transform(df_pred_input[col])

df_all['PREDICTED_ARR_DELAY'] = rf.predict(df_pred_input)

graph = defaultdict(dict)
route_times = defaultdict(dict)
predicted_delay_df = df_all.groupby(['ORIGIN_CITY', 'DEST_CITY']).agg({
    'PREDICTED_ARR_DELAY': 'mean',
    'AIR_TIME': 'mean'
}).reset_index()

for _, row in predicted_delay_df.iterrows():
    origin, dest = row['ORIGIN_CITY'], row['DEST_CITY']
    cost = row['PREDICTED_ARR_DELAY'] + row['AIR_TIME']
    graph[origin][dest] = cost
    route_times[origin][dest] = (round(row['PREDICTED_ARR_DELAY'], 2), round(row['AIR_TIME'], 2))

def recommend_two_leg_path(graph, origin, destination):
    if destination not in graph.get(origin, {}):
        return "No direct route found."

    direct_cost = graph[origin][destination]
    best_path = None
    best_total_cost = float('inf')

    for mid in graph[origin]:
        if mid == destination or destination not in graph.get(mid, {}):
            continue
        total_cost = graph[origin][mid] + graph[mid][destination]
        if total_cost <= direct_cost and total_cost < best_total_cost:
            best_total_cost = total_cost
            best_path = (origin, mid, destination)

    if best_path:
        return {
            'direct_predicted_delay': route_times[origin][destination][0],
            'direct_air_time': route_times[origin][destination][1],
            'recommended_path': best_path,
            'combined_predicted_delay': round(route_times[best_path[0]][best_path[1]][0] + route_times[best_path[1]][best_path[2]][0], 2),
            'combined_air_time': round(route_times[best_path[0]][best_path[1]][1] + route_times[best_path[1]][best_path[2]][1], 2)
        }
    else:
        return "No better 2-leg path found."

def visualize_route(route_dict):
    if isinstance(route_dict, str):
        print(route_dict)
        return
    G = nx.DiGraph()
    path = route_dict['recommended_path']
    delay1, time1 = route_times[path[0]][path[1]]
    delay2, time2 = route_times[path[1]][path[2]]
    
    G.add_edge(path[0], path[1], weight=delay1 + time1, label=f"{delay1}+{time1}")
    G.add_edge(path[1], path[2], weight=delay2 + time2, label=f"{delay2}+{time2}")
    
    pos = nx.spring_layout(G, seed=42)
    edge_labels = nx.get_edge_attributes(G, 'label')
    
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Recommended Two-Leg Flight Path (Delay + Air Time)")
    plt.show()


# Examples
example1 = recommend_two_leg_path(graph, 'Phoenix, AZ', 'New York, NY')
print("\nExample 1:", example1)
visualize_route(example1)

example2 = recommend_two_leg_path(graph, 'Chicago, IL', 'Los Angeles, CA')
print("\nExample 2:", example2)
visualize_route(example2)
