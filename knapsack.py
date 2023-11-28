# Copyright 2021 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#pip install streamlit
#pip install plotly-express

# streamlit run /workspaces/knapsack/knapsack.py



import os
import itertools
import click
import pandas as pd
from dwave.system import LeapHybridCQMSampler
from dimod import ConstrainedQuadraticModel, BinaryQuadraticModel, QuadraticModel  
import pickle
import plotly.express as px
import time

def parse_inputs(data_file, capacity):
    """Parse user input and files for data to build CQM.

    Args:
        data_file (csv file):
            File of items (weight & cost) slated to ship.
        capacity (int):
            Max weight the shipping container can accept.

    Returns:
        Costs, weights, and capacity.
    """
    df = pd.read_csv(data_file, names=['cost', 'weight'])

    if not capacity:
        capacity = int(0.8 * sum(df['weight']))
        print("\nSetting weight capacity to 80% of total: {}".format(str(capacity)))

    return df['cost'], df['weight'], capacity

def build_knapsack_cqm(costs, weights, max_weight):
    """Construct a CQM for the knapsack problem.

    Args:
        costs (array-like):
            Array of costs for the items.
        weights (array-like):
            Array of weights for the items.
        max_weight (int):
            Maximum allowable weight for the knapsack.

    Returns:
        Constrained quadratic model instance that represents the knapsack problem.
    """
    num_items = len(costs)
    print("\nBuilding a CQM for {} items.".format(str(num_items)))

    cqm = ConstrainedQuadraticModel()
    obj = BinaryQuadraticModel(vartype='BINARY')
    constraint = QuadraticModel()

    for i in range(num_items):
        # Objective is to maximize the total costs
        obj.add_variable(i)
        obj.set_linear(i, -costs[i])
        # Constraint is to keep the sum of items' weights under or equal capacity
        constraint.add_variable('BINARY', i)
        constraint.set_linear(i, weights[i])

    cqm.set_objective(obj)
    cqm.add_constraint(constraint, sense="<=", rhs=max_weight, label='capacity')

    return cqm

def parse_solution(sampleset, costs, weights):
    """Translate the best sample returned from solver to shipped items.

    Args:

        sampleset (dimod.Sampleset):
            Samples returned from the solver.
        costs (array-like):
            Array of costs for the items.
        weights (array-like):
            Array of weights for the items.
    """
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    if not len(feasible_sampleset):
        raise ValueError("No feasible solution found")

    best = feasible_sampleset.first

    selected_item_indices = [key for key, val in best.sample.items() if val==1.0]
    selected_weights = list(weights.loc[selected_item_indices])
    selected_costs = list(costs.loc[selected_item_indices])

    # save results for testing
    # Open a file in binary write mode
    with open('selected_weights.pkl', 'wb') as file:
        pickle.dump(selected_weights, file)
    
    with open('selected_costs.pkl', 'wb') as file:
        pickle.dump(selected_costs, file)

    print("\nFound best solution at energy {}".format(best.energy))
    print("\nSelected item numbers (0-indexed):", selected_item_indices)
    print("\nSelected item weights: {}, total = {}".format(selected_weights, sum(selected_weights)))
    print("\nSelected item costs: {}, total = {}".format(selected_costs, sum(selected_costs)))

    return selected_item_indices, selected_weights, selected_costs

def datafile_help(max_files=5):
    """Provide content of input file names and total weights for click()'s --help."""

    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        datafiles = os.listdir(data_dir)
        # "\b" enables newlines in click() help text
        help = """
\b
Name of data file (under the 'data/' folder) to run on.
One of:
File Name \t Total weight
"""
        for file in datafiles[:max_files]:
            _, weights, _ = parse_inputs(os.path.join(data_dir, file), 1234)
            help += "{:<20} {:<10} \n".format(str(file), str(sum(weights)))
        help += "\nDefault is to run on data/large.csv."
    except:
        help = """
\b
Name of data file (under the 'data/' folder) to run on.
Default is to run on data/large.csv.
"""

    return help

filename_help = datafile_help()     # Format the help string for the --filename argument


def runSolver(filename, capacity):
    """Solve a knapsack problem using a CQM solver and return the selected items."""

    sampler = LeapHybridCQMSampler()

    costs, weights, capacity = parse_inputs(filename, capacity)

    cqm = build_knapsack_cqm(costs, weights, capacity)

    print("Submitting CQM to solver {}.".format(sampler.solver.name))
    sampleset = sampler.sample_cqm(cqm, label='Example - Knapsack')

    selected_item_indices, selected_weights, selected_costs = parse_solution(sampleset, costs, weights)

    return selected_item_indices, selected_weights, selected_costs



#runSolver()

# Open the file in binary read mode
with open('selected_weights.pkl', 'rb') as file:
    selected_weights = pickle.load(file)

with open('selected_costs.pkl', 'rb') as file:
    selected_costs = pickle.load(file)  

print(selected_weights) 
print(selected_costs)



## UI Section

import streamlit as st
import pandas as pd
import os

import streamlit as st

st.set_page_config(layout="wide")

# Page title and description
st.title("Asset Optimization Dashboard")
st.markdown("""
This application utilizes Weighted Goal Programming (WGP) to optimize asset allocation.
It helps in making informed decisions about resource allocation by balancing multiple goals,
such as budget adherence, procurement efficiency, and capability enhancement.  The goal: Integration of Informational
and Physical Power.
""")


# Custom CSS for styling
st.markdown("""
<style>
    /* Import Fira Code Font */
    @import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

    /* Apply Fira Code Font to the entire app */
    html, body, [class*="css"] {
        font-family: 'Fira Code', monospace;
    }

    /* Dark Background */
    .css-18e3th9 {
        background-color: #1e1e1e; /* Dark background color */
        color: #ffffff; /* Light text color for contrast */
    }

    /* Adjusting Title Color */
    h1 {
        color: #4CAF50; /* Modern green */
    }

    /* Custom style for buttons */
    .stButton>button {
        color: #ffffff;
        background-color: #007BFF; /* Bright blue for buttons */
        border-radius: 5px;
        border: none; /* Remove default border */
    }

    /* Table styling for dark theme */
    .stDataFrame {
        background-color: #2e2e2e; /* Slightly lighter than the main background */
        color: #ffffff;
    }

    /* Additional styling can go here */

</style>
""", unsafe_allow_html=True)

# File names
file_names = ['very_small.csv', 'small.csv', 'large.csv', 'very_large.csv', 'huge.csv']

# Path to the data folder
data_folder = 'data'

# Add a dropdown in the sidebar to select the file
selected_file = st.sidebar.selectbox("Select an example file", file_names)

# Full path to the selected file
file_path = os.path.join(data_folder, selected_file)

if os.path.exists(file_path):
    # Read file without headers
    data = pd.read_csv(file_path, header=None, names=['Value', 'Cost'])

    # Create an 'Item' column with sequential numbers
    data['Item'] = range(1, len(data) + 1)

    # Reorder the columns to 'Item', 'Value', 'Cost'
    data = data[['Item', 'Value', 'Cost']]

    # Calculate 75% of the sum of the Cost values
    seventy_five_percent_cost = round(0.75 * data['Cost'].sum())

    # Display in the sidebar
    st.sidebar.markdown("### 75% Default")
    st.sidebar.text(f"{seventy_five_percent_cost}")  # Display with 2 decimal places

    max_weight = st.number_input("Enter the maximum weight", value=seventy_five_percent_cost)

    if st.button('Solve WGP Model'):

        start_time = time.time()

        try:
            selected_indices, selected_values, selected_costs = runSolver(file_path, max_weight)

            # Calculate and display runtime
            runtime = time.time() - start_time  # End time after calling runSolver
            st.sidebar.markdown("### Runtime of Solver")
            st.sidebar.text(f"{runtime:.2f} seconds")  # Display runtime in seconds

            # Create a DataFrame for the selected items
            output_df = pd.DataFrame({
                'Item Index': selected_indices,
                'Value': selected_values,
                'Cost': selected_costs
            })

            st.subheader('Optimization Results')

            data['parents'] = ''  # This creates a column with empty strings, meaning no parent node


            # Correctly mark selected items
            data['IsSelected'] = data.index.isin(selected_indices)

            # Creating a 'Label' column for displaying text on the treemap
            data['Label'] = data['Cost'].astype(str)

            # Create the treemap
            fig_selected = px.treemap(
                data,
                path=['parents', 'Item'],  # Use the new 'parents' column here
                values='Value',  # Size of the box based on 'Value'
                color='IsSelected',  # Color based on selection status
                color_discrete_map={True: 'darkgreen', False: 'darkred'},  # Color mapping
                title="Selected Items: Highlighted",
            )

            fig_selected.update_traces(
                textinfo="label+text+value",
                texttemplate='%{label}<br>%{customdata[0]}',  # Custom text template
                customdata=data[['Label']],  # Data to be used in the text template
                hovertemplate="<b>%{label}</b><br>Cost: %{customdata[0]}<br>Value: %{value}<extra></extra>",
                marker=dict(line=dict(width=2, color='rgba(48, 48, 48, 1)'))  # Set line color to dark gray

            )

            fig_selected.update_layout(margin=dict(t=50, l=25, r=25, b=25), width=800, height=800)
            

            st.plotly_chart(fig_selected)

            st.write('The following table displays the selected items, their values, and costs:')
            st.markdown(output_df.to_html(index=False), unsafe_allow_html=True)

        except ValueError as e:
            st.error(str(e))




else:
    st.error(f"File not found: {file_path}")




