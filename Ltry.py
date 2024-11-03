import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import networkx as nx
from plotly.offline import plot
import webbrowser
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from keras.models import Sequential
from keras.layers import Input, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from keras.optimizers import Adam
from sklearn.cluster import DBSCAN
import io
import sys
import tensorflow as tf
import kmapper as km  # Add kmapper for creating Mapper graphs

# Ensure TensorFlow settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def preprocess_data(data, draw_columns, option='Option 1'):
    if 'Draw Date' in data.columns:
        data['Draw Date'] = pd.to_datetime(data['Draw Date'], errors='coerce')
        data['Year'] = data['Draw Date'].dt.year
        data['Month'] = data['Draw Date'].dt.month
        data['Day'] = data['Draw Date'].dt.day
        data = data.drop(columns=['Draw Date'])
    else:
        print("Warning: 'Draw Date' column not found. Skipping date preprocessing.")

    categorical_columns = data.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_columns:
        if col not in draw_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le

    if option == 'Option 1':
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    elif option == 'Option 2':
        columns_to_impute = [col for col in data.columns if col not in draw_columns]
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data[columns_to_impute]), columns=columns_to_impute,
                                    index=data.index)
        data_imputed = pd.concat([data[draw_columns], data_imputed], axis=1)
    else:
        raise ValueError(f"Unknown preprocessing option: {option}")

    return data_imputed, label_encoders


def create_tda_graph(csv_file, output_html='tda_graph.html'):
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Create a simple graph using NetworkX
    G = nx.Graph()

    # Example of adding nodes and edges based on data
    for i in range(len(data) - 1):
        G.add_node(i, label=str(data.iloc[i]['Draw 1']))  # Use 'Draw 1' as node labels
        if i > 0:
            G.add_edge(i-1, i)  # Connect consecutive nodes

    # Convert NetworkX graph to Plotly graph
    pos = nx.spring_layout(G)
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([G.nodes[node]['label']])

    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title='<br>TDA Graph Representation',
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
                    ))

    # Save the figure as an HTML file
    plot(fig, filename=output_html)
    print(f"TDA graph saved as {output_html}")

    return output_html


def run_tda_analysis(csv_file, draw_columns, model_type='Gradient Boosting', preprocess_func=None):
    output = io.StringIO()  # Capture output in a StringIO object
    sys.stdout = output  # Redirect stdout to the StringIO object

    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return output.getvalue()

    if not all(col in data.columns for col in draw_columns):
        print("Error: Missing expected columns in CSV file.")
        return output.getvalue()

    data, label_encoders = preprocess_func(data)  # Call preprocess_func with only the data

    if data.shape[1] <= len(draw_columns):
        print("Error: Not enough columns for analysis.")
        return output.getvalue()

    X = data.drop(columns=draw_columns)
    y = data[draw_columns].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'Gradient Boosting':
        model = MultiOutputRegressor(GradientBoostingRegressor())
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error of the Gradient Boosting model: {mse}")

    elif model_type == 'Neural Network':
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(draw_columns)))
        model.compile(optimizer=Adam(), loss='mse')
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)  # Suppress epoch output
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error of the Neural Network model: {mse}")

    else:
        print(f"Error: Unsupported model type {model_type}.")
        return output.getvalue()

    if X.empty:
        print("Error: No recent data available for prediction.")
        return output.getvalue()

    recent_data = pd.DataFrame(X.tail(1), columns=X.columns)
    next_draw_predictions = model.predict(recent_data)
    next_draw_predictions_rounded = np.round(next_draw_predictions).astype(int)

    next_draw_df = pd.DataFrame(next_draw_predictions_rounded, columns=draw_columns)
    print(f"Predicted next draw numbers:\n{next_draw_df}")

    sys.stdout = sys.__stdout__  # Reset stdout to default
    return output.getvalue()

def create_mapper_graph(csv_file, output_html='mapper_output.html'):
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

    # Preprocessing for Mapper graph (optional, you can skip if already preprocessed)
    data_imputed, _ = preprocess_data(data, draw_columns=[])  # Adjust preprocessing if necessary

    if not isinstance(data_imputed, pd.DataFrame):
        print("Error: The imputed data is not a DataFrame.")
        return None

    print(f"Data shape for Mapper: {data_imputed.shape}")

    # Initialize KeplerMapper
    mapper = km.KeplerMapper()

    try:
        # Use a custom projection function
        def custom_projection(X):
            return X.sum(axis=1).values.reshape(-1, 1)

        # Apply projection manually to see the result
        print("Applying custom projection...")
        projected_data = custom_projection(data_imputed)
        print(f"Projected data shape: {projected_data.shape}")

        # Fit and transform the data to the lens
        lens = mapper.fit_transform(projected_data)
        print(f"Lens shape: {lens.shape}")

        # Create a simplicial complex with custom clustering parameters
        clusterer = DBSCAN(eps=0.5, min_samples=1)
        graph = mapper.map(lens, data_imputed, clusterer=clusterer)

        # Visualize the simplicial complex with Plotly
        mapper.visualize(graph, path_html=output_html)
        print(f"Mapper graph saved as {output_html}")

    except Exception as e:
        print(f"Error during Mapper processing: {e}")
        return None

    return output_html


class TDAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TDA Analysis")

        # Set a modern theme
        self.style = ttk.Style()
        self.style.configure('TButton', padding=6, relief='flat', background='#4CAF50', foreground='white')
        self.style.configure('TRadioButton', padding=6, background='#ffffff')
        self.style.configure('TLabel', background='#ffffff', font=('Arial', 12))

        # Custom style for the "Run Analysis" button
        self.style.configure('RunButton.TButton', background='#007bff', foreground='black', font=('Arial', 12, 'bold'))
        self.style.map('RunButton.TButton', background=[('active', '#0056b3')])

        # Data file
        self.data_file = tk.StringVar()

        # Create widgets
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Select Dataset:", font=('Arial', 14)).grid(row=0, column=0, padx=10, pady=10,
                                                                              sticky='w')
        self.dataset_choice = tk.StringVar(value='Mega Millions')
        ttk.Radiobutton(self.root, text="Mega Millions", variable=self.dataset_choice, value='Mega Millions').grid(
            row=0, column=1, padx=10, pady=10, sticky='w')
        ttk.Radiobutton(self.root, text="Powerball", variable=self.dataset_choice, value='Powerball').grid(row=0,
                                                                                                           column=2,
                                                                                                           padx=10,
                                                                                                           pady=10,
                                                                                                           sticky='w')
        ttk.Radiobutton(self.root, text="Florida Lotto", variable=self.dataset_choice, value='LPlay').grid(row=0,
                                                                                                           column=3,
                                                                                                           padx=10,
                                                                                                           pady=10,
                                                                                                           sticky='w')

        ttk.Label(self.root, text="Select model:", font=('Arial', 14)).grid(row=1, column=0, padx=10, pady=10,
                                                                            sticky='w')
        self.model_choice = tk.StringVar(value='Gradient Boosting')
        ttk.Radiobutton(self.root, text="Gradient Boosting", variable=self.model_choice,
                        value='Gradient Boosting').grid(row=1, column=1, padx=10, pady=10, sticky='w')
        ttk.Radiobutton(self.root, text="Neural Network", variable=self.model_choice, value='Neural Network').grid(
            row=1, column=2, padx=10, pady=10, sticky='w')

        ttk.Label(self.root, text="Select preprocessing option:", font=('Arial', 14)).grid(row=2, column=0, padx=10,
                                                                                           pady=10, sticky='w')
        self.preprocess_option = tk.StringVar(value='Option 1')
        ttk.Radiobutton(self.root, text="Option 1 (Mega & Lotto)", variable=self.preprocess_option,
                        value='Option 1').grid(row=2, column=1, padx=10, pady=10, sticky='w')
        ttk.Radiobutton(self.root, text="Option 2 (PowerBall)", variable=self.preprocess_option, value='Option 2').grid(
            row=2, column=2, padx=10, pady=10, sticky='w')

        ttk.Button(self.root, text="Run Analysis", style='RunButton.TButton', command=self.run_analysis).grid(row=3,
                                                                                                              column=0,
                                                                                                              columnspan=4,
                                                                                                              padx=10,
                                                                                                              pady=10)

        # Button to create and view TDA graph
        ttk.Button(self.root, text="Create TDA Graph", style='RunButton.TButton', command=self.create_and_view_tda_graph).grid(row=4,
                                                                                                              column=0,
                                                                                                              columnspan=4,
                                                                                                              padx=10,
                                                                                                              pady=10)

        # Add a button for creating and viewing the Mapper graph
        ttk.Button(self.root, text="Create Mapper Graph", style='RunButton.TButton',
                   command=self.create_and_view_mapper_graph).grid(row=5, column=0, columnspan=4, padx=10, pady=10)

        # Message label
        self.message_label = ttk.Label(self.root, text="Welcome Taiwo. Pls make sure the csv file data are up to date",
                                       font=('Arial', 12), foreground='blue')
        self.message_label.grid(row=6, column=0, columnspan=4, padx=10, pady=10)

        self.result_text = tk.Text(self.root, width=100, height=20, wrap='word', font=('Arial', 12))
        self.result_text.grid(row=7, column=0, columnspan=4, padx=10, pady=10)


    def run_analysis(self):
        self.message_label.config(text="Running analysis...")  # Update message label with status

        dataset = self.dataset_choice.get()
        if not dataset:
            messagebox.showerror("Error", "Please select a dataset.")
            self.message_label.config(text="Error: Please select a dataset.")
            return

        dataset_map = {
            'Mega Millions': (
            'Mega Million Past Winning Numbers.csv', ['Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'MegaBall']),
            'Powerball': (
            'PowerBall Past Winning Numbers.csv', ['Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Power']),
            'LPlay': (
            'Florida Lotto Past Winning Numbers.csv', ['Draw 1', 'Draw 2', 'Draw 3', 'Draw 4', 'Draw 5', 'Draw 6'])
        }

        if dataset not in dataset_map:
            messagebox.showerror("Error", "Invalid dataset selected.")
            self.message_label.config(text="Error: Invalid dataset selected.")
            return

        csv_file, draw_columns = dataset_map[dataset]
        preprocess_option = self.preprocess_option.get()

        preprocess_func = lambda data: preprocess_data(data, draw_columns, preprocess_option)
        model_type = self.model_choice.get()

        output = run_tda_analysis(csv_file, draw_columns, model_type=model_type, preprocess_func=preprocess_func)

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, output)
        self.message_label.config(text="Analysis completed.")  # Update message label with status

    def create_and_view_tda_graph(self):
        dataset = self.dataset_choice.get()
        dataset_map = {
            'Mega Millions': 'Mega Million Past Winning Numbers.csv',
            'Powerball': 'PowerBall Past Winning Numbers.csv',
            'LPlay': 'Florida Lotto Past Winning Numbers.csv'
        }

        if dataset not in dataset_map:
            messagebox.showerror("Error", "Invalid dataset selected.")
            return

        csv_file = dataset_map[dataset]
        output_html = create_tda_graph(csv_file)

        if output_html:
            webbrowser.open(output_html)
        else:
            messagebox.showerror("Error", "Failed to create TDA graph.")


    def create_and_view_mapper_graph(self):
        dataset = self.dataset_choice.get()
        dataset_map = {
            'Mega Millions': 'Mega Million Past Winning Numbers.csv',
            'Powerball': 'PowerBall Past Winning Numbers.csv',
            'LPlay': 'Florida Lotto Past Winning Numbers.csv'
        }

        if dataset not in dataset_map:
            messagebox.showerror("Error", "Invalid dataset selected.")
            return

        csv_file = dataset_map[dataset]
        output_html = create_mapper_graph(csv_file)

        if output_html:
            webbrowser.open(output_html)
        else:
            messagebox.showerror("Error", "Failed to create Mapper graph.")


if __name__ == "__main__":
    root = tk.Tk()
    app = TDAApp(root)
    root.mainloop()

