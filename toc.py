# Required Libraries
import streamlit as st
import pandas as pd
import networkx as nx
from pyvis.network import Network
# import other libraries like pdfminer or PyPDF2 for PDF processing, graphviz, plotly, bokeh etc.

# Function to extract Table of Contents from PDF
def extract_toc_from_pdf(file):
    # Use pdfminer or PyPDF2 to extract text from PDF
    # Assume the function returns the TOC as a list of strings
    # For this example, let's assume it returns a dummy list
    return ["Chapter 1: Introduction", "1.1 Background", "1.2 Objectives", "Chapter 2: Methodology"]

# Function to create network diagram from TOC
def create_network_from_toc(toc_list):
    # Create a network graph based on hierarchy in TOC
    # Returns the network graph
    # This is a stub, in real application you'd parse the TOC to build the graph
    G = nx.DiGraph()
    G.add_edges_from([("Chapter 1: Introduction", "1.1 Background"), ("Chapter 1: Introduction", "1.2 Objectives")])
    return G

# Streamlit App
st.title('ToC Diagrammer')

# Sidebar
st.sidebar.header('Upload Table of Contents')
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        toc_list = extract_toc_from_pdf(uploaded_file)
    else:  # handle other file types
        toc_list = pd.read_csv(uploaded_file).values.tolist()

    # Display summary stats
    st.sidebar.text(f'Number of Words: {sum(len(item.split()) for item in toc_list)}')
    st.sidebar.text(f'Max Hierarchical Levels: {max(item.count(".") for item in toc_list)}')

# Main Display
toc_graph = create_network_from_toc(toc_list)
net = Network(notebook=True)
net.from_nx(toc_graph)
net.show("tmp.html")
# st.write(toc_graph)
st.components.v1.html(open("tmp.html", "r").read(), width=800, height=800)


# Display Dataframe
df = pd.DataFrame(toc_list, columns=["ToC Item"])
st.write(df)

# Graph Statistics
st.write(f'Number of Nodes: {toc_graph.number_of_nodes()}')
st.write(f'Number of Edges: {toc_graph.number_of_edges()}')

# Download as CSV
st.markdown('<a href="data:file/csv;base64,{}" download="toc_data.csv">Download TOC Data</a>'
            .format(pd.DataFrame(toc_list).to_csv(index=False, encoding='utf-8')), unsafe_allow_html=True)
