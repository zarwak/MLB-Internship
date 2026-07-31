"""
Streamlit App: Iris Flower Clustering & Visualization
Interactive web application for clustering and PCA analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Iris Clustering & PCA Visualization",
    page_icon="🌺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2C3E50;
        padding: 0.5rem;
        border-bottom: 2px solid #3498db;
        margin-bottom: 1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 5px;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.feature_names = None
    st.session_state.target_names = None
    st.session_state.X_scaled = None
    st.session_state.clusters = None
    st.session_state.kmeans = None
    st.session_state.X_pca = None
    st.session_state.pca = None
    st.session_state.df = None

# Title
st.markdown('<div class="main-header">🌺 Iris Flower Clustering & PCA Visualization</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Data Source")
    
    data_source = st.radio(
        "Select data source:",
        ["Use Iris Dataset", "Upload CSV File"]
    )
    
    if data_source == "Use Iris Dataset":
        if st.button("Load Iris Dataset"):
            with st.spinner("Loading Iris dataset..."):
                iris = load_iris()
                df = pd.DataFrame(iris.data, columns=iris.feature_names)
                df['species'] = iris.target
                df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                
                st.session_state.data = iris.data
                st.session_state.feature_names = iris.feature_names
                st.session_state.target_names = iris.target_names
                st.session_state.df = df
                st.session_state.y = iris.target
                
                # Scale data
                scaler = StandardScaler()
                st.session_state.X_scaled = scaler.fit_transform(iris.data)
                
                st.success("✅ Iris dataset loaded successfully!")
                st.info(f"📊 Shape: {iris.data.shape}")
    
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
                # Show preview
                st.dataframe(df.head())
                
                # Select features
                st.markdown("### 🎯 Select Features")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if len(numeric_cols) >= 2:
                    feature_cols = st.multiselect(
                        "Select feature columns:",
                        numeric_cols,
                        default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
                    )
                    
                    if feature_cols:
                        if st.button("Process Data"):
                            X = df[feature_cols].values
                            st.session_state.data = X
                            st.session_state.feature_names = feature_cols
                            st.session_state.df = df
                            st.session_state.X_scaled = StandardScaler().fit_transform(X)
                            st.success("✅ Data processed successfully!")
                else:
                    st.warning("Please upload a dataset with at least 2 numeric columns.")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    st.markdown("---")
    st.markdown("### ⚙️ Model Parameters")
    
    # K-Means parameters
    n_clusters = st.slider("Number of Clusters (k)", 2, 8, 3)
    
    if st.button("🔄 Run Analysis", type="primary"):
        if st.session_state.X_scaled is not None:
            with st.spinner("Running analysis..."):
                # K-Means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(st.session_state.X_scaled)
                st.session_state.clusters = clusters
                st.session_state.kmeans = kmeans
                
                # PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(st.session_state.X_scaled)
                st.session_state.X_pca = X_pca
                st.session_state.pca = pca
                
                st.success("✅ Analysis complete!")
        else:
            st.warning("Please load data first!")
    
    st.markdown("---")
    st.markdown("### 📖 About")
    st.info("""
    This app demonstrates:
    - **K-Means Clustering** for grouping similar data points
    - **PCA** for dimensionality reduction
    - Interactive visualizations
    - Upload custom datasets
    """)

# Main content
if st.session_state.X_scaled is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Samples", len(st.session_state.X_scaled))
    with col2:
        st.metric("Features", st.session_state.X_scaled.shape[1])
    with col3:
        if st.session_state.clusters is not None:
            st.metric("Clusters", len(np.unique(st.session_state.clusters)))
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Exploration", "🎯 K-Means Clustering", "📉 PCA Visualization", "📈 Comparison"])
    
    with tab1:
        st.markdown('<div class="sub-header">📊 Dataset Exploration</div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Data Preview")
                st.dataframe(st.session_state.df.head(20))
            
            with col2:
                st.subheader("Summary Statistics")
                if 'species_name' in st.session_state.df.columns:
                    st.write(st.session_state.df.describe())
            
            # Visualizations
            st.subheader("Feature Distributions")
            
            fig = make_subplots(rows=2, cols=2, subplot_titles=st.session_state.feature_names[:4])
            
            for i, feature in enumerate(st.session_state.feature_names[:4]):
                row, col = i // 2 + 1, i % 2 + 1
                if 'species_name' in st.session_state.df.columns:
                    for species in st.session_state.df['species_name'].unique():
                        subset = st.session_state.df[st.session_state.df['species_name'] == species]
                        fig.add_trace(
                            go.Histogram(x=subset[feature], name=species, 
                                        nbinsx=15, opacity=0.6, legendgroup=species),
                            row=row, col=col
                        )
                else:
                    fig.add_trace(
                        go.Histogram(x=st.session_state.df[feature], nbinsx=15),
                        row=row, col=col
                    )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix
            st.subheader("Feature Correlation Matrix")
            corr = st.session_state.df[st.session_state.feature_names].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", 
                          color_continuous_scale="RdBu", range_color=[-1, 1])
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown('<div class="sub-header">🎯 K-Means Clustering Results</div>', unsafe_allow_html=True)
        
        if st.session_state.clusters is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Cluster Distribution")
                cluster_counts = pd.Series(st.session_state.clusters).value_counts().sort_index()
                fig = px.bar(x=cluster_counts.index, y=cluster_counts.values,
                           labels={'x': 'Cluster', 'y': 'Count'},
                           title="Number of Samples per Cluster",
                           color=cluster_counts.index,
                           color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Clustering Metrics")
                if hasattr(st.session_state, 'y') and st.session_state.y is not None:
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(st.session_state.y, st.session_state.clusters)
                    st.metric("Adjusted Rand Index", f"{ari:.3f}")
                
                sil_score = silhouette_score(st.session_state.X_scaled, st.session_state.clusters)
                st.metric("Silhouette Score", f"{sil_score:.3f}")
                
                inertia = st.session_state.kmeans.inertia_
                st.metric("Inertia (WCSS)", f"{inertia:.1f}")
            
            # 2D Cluster Visualization
            st.subheader("Cluster Visualization (First Two Features)")
            
            # Create 2D scatter plot
            fig = go.Figure()
            
            # Get first two features
            feature1 = st.selectbox("Select X-axis feature:", st.session_state.feature_names, key="cluster_x")
            feature2 = st.selectbox("Select Y-axis feature:", st.session_state.feature_names, key="cluster_y", 
                                  index=min(1, len(st.session_state.feature_names)-1))
            
            idx1 = st.session_state.feature_names.index(feature1)
            idx2 = st.session_state.feature_names.index(feature2)
            
            if 'species_name' in st.session_state.df.columns:
                # Color by cluster
                for cluster in np.unique(st.session_state.clusters):
                    mask = st.session_state.clusters == cluster
                    fig.add_trace(go.Scatter(
                        x=st.session_state.X_scaled[mask, idx1],
                        y=st.session_state.X_scaled[mask, idx2],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8, opacity=0.7)
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=st.session_state.X_scaled[:, idx1],
                    y=st.session_state.X_scaled[:, idx2],
                    mode='markers',
                    marker=dict(color=st.session_state.clusters, 
                              colorscale='Viridis', size=8, opacity=0.7)
                ))
            
            # Add centroids
            if st.session_state.kmeans is not None:
                centroids = st.session_state.kmeans.cluster_centers_
                fig.add_trace(go.Scatter(
                    x=centroids[:, idx1],
                    y=centroids[:, idx2],
                    mode='markers',
                    name='Centroids',
                    marker=dict(symbol='x', size=15, color='red', line_width=2)
                ))
            
            fig.update_layout(
                title=f'K-Means Clusters (k={len(np.unique(st.session_state.clusters))})',
                xaxis_title=f'{feature1} (scaled)',
                yaxis_title=f'{feature2} (scaled)',
                height=500,
                hovermode='closest'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 3D Cluster Visualization
            st.subheader("3D Cluster Visualization")
            if len(st.session_state.feature_names) >= 3:
                features_3d = st.multiselect(
                    "Select 3 features for 3D visualization:",
                    st.session_state.feature_names,
                    default=st.session_state.feature_names[:3],
                    max_selections=3,
                    key="features_3d"
                )
                
                if len(features_3d) == 3:
                    idxs = [st.session_state.feature_names.index(f) for f in features_3d]
                    fig = go.Figure()
                    
                    for cluster in np.unique(st.session_state.clusters):
                        mask = st.session_state.clusters == cluster
                        fig.add_trace(go.Scatter3d(
                            x=st.session_state.X_scaled[mask, idxs[0]],
                            y=st.session_state.X_scaled[mask, idxs[1]],
                            z=st.session_state.X_scaled[mask, idxs[2]],
                            mode='markers',
                            name=f'Cluster {cluster}',
                            marker=dict(size=5, opacity=0.7)
                        ))
                    
                    fig.update_layout(
                        title='3D Cluster Visualization',
                        scene=dict(
                            xaxis_title=features_3d[0],
                            yaxis_title=features_3d[1],
                            zaxis_title=features_3d[2]
                        ),
                        height=600
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<div class="sub-header">📉 PCA Visualization</div>', unsafe_allow_html=True)
        
        if st.session_state.X_pca is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PCA Results (2D)")
                
                fig = go.Figure()
                
                if 'species_name' in st.session_state.df.columns:
                    for species in st.session_state.df['species_name'].unique():
                        mask = st.session_state.df['species_name'] == species
                        fig.add_trace(go.Scatter(
                            x=st.session_state.X_pca[mask, 0],
                            y=st.session_state.X_pca[mask, 1],
                            mode='markers',
                            name=species,
                            marker=dict(size=10, opacity=0.7)
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=st.session_state.X_pca[:, 0],
                        y=st.session_state.X_pca[:, 1],
                        mode='markers',
                        marker=dict(color=st.session_state.clusters, 
                                  colorscale='Viridis', size=10, opacity=0.7)
                    ))
                
                fig.update_layout(
                    title='PCA - 2D Projection',
                    xaxis_title=f'PC1 ({st.session_state.pca.explained_variance_ratio_[0]:.1%})',
                    yaxis_title=f'PC2 ({st.session_state.pca.explained_variance_ratio_[1]:.1%})',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Explained Variance")
                
                # Explained variance
                explained_var = st.session_state.pca.explained_variance_ratio_
                cumulative_var = np.cumsum(explained_var)
                
                fig = make_subplots(rows=2, cols=1, subplot_titles=("Individual Variance", "Cumulative Variance"))
                
                fig.add_trace(
                    go.Bar(x=[f'PC{i+1}' for i in range(len(explained_var))], 
                          y=explained_var, name='Individual'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=[f'PC{i+1}' for i in range(len(cumulative_var))], 
                              y=cumulative_var, name='Cumulative', 
                              mode='lines+markers'),
                    row=2, col=1
                )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature loadings
            st.subheader("Feature Loadings on Principal Components")
            loadings = pd.DataFrame(
                st.session_state.pca.components_.T,
                columns=[f'PC{i+1}' for i in range(st.session_state.pca.n_components_)],
                index=st.session_state.feature_names
            )
            
            fig = px.imshow(loadings, text_auto=True, aspect="auto",
                          color_continuous_scale="RdBu", range_color=[-1, 1])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Explained variance table
            variance_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(len(explained_var))],
                'Explained Variance Ratio': explained_var,
                'Cumulative Variance': cumulative_var
            })
            st.dataframe(variance_df.style.format({'Explained Variance Ratio': '{:.2%}', 'Cumulative Variance': '{:.2%}'}))
    
    with tab4:
        st.markdown('<div class="sub-header">📈 Comparison & Insights</div>', unsafe_allow_html=True)
        
        if st.session_state.clusters is not None and st.session_state.X_pca is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("PCA with Clusters")
                fig = go.Figure()
                
                for cluster in np.unique(st.session_state.clusters):
                    mask = st.session_state.clusters == cluster
                    fig.add_trace(go.Scatter(
                        x=st.session_state.X_pca[mask, 0],
                        y=st.session_state.X_pca[mask, 1],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(size=8, opacity=0.7)
                    ))
                
                fig.update_layout(
                    title='PCA with K-Means Clusters',
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Dimensionality Reduction Analysis")
                
                original_dim = st.session_state.X_scaled.shape[1]
                pca_dim = 2
                
                st.metric("Original Dimensions", original_dim)
                st.metric("PCA Dimensions", pca_dim)
                st.metric("Reduction", f"{original_dim - pca_dim} ({((original_dim - pca_dim)/original_dim*100):.0f}%)")
                st.metric("Variance Preserved", f"{sum(st.session_state.pca.explained_variance_ratio_):.1%}")
            
            # Side by side comparison
            st.subheader("Original vs PCA Visualization")
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Original Data (First 2 Features)", "PCA Visualization"))
            
            # Original data
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.X_scaled[:, 0],
                    y=st.session_state.X_scaled[:, 1],
                    mode='markers',
                    marker=dict(color=st.session_state.clusters, colorscale='Viridis', size=8, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # PCA data
            fig.add_trace(
                go.Scatter(
                    x=st.session_state.X_pca[:, 0],
                    y=st.session_state.X_pca[:, 1],
                    mode='markers',
                    marker=dict(color=st.session_state.clusters, colorscale='Viridis', size=8, opacity=0.7),
                    showlegend=False
                ),
                row=1, col=2
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Insights
            st.subheader("💡 Key Insights")
            
            insight_cols = st.columns(3)
            
            with insight_cols[0]:
                st.markdown("""
                <div class="info-box">
                    <strong>🎯 Clustering Insights</strong>
                    <ul>
                        <li>K-Means successfully identified {} clusters</li>
                        <li>Silhouette Score: {:.3f}</li>
                        <li>Cluster distribution: {}
                    </ul>
                </div>
                """.format(
                    len(np.unique(st.session_state.clusters)),
                    silhouette_score(st.session_state.X_scaled, st.session_state.clusters),
                    ', '.join([f"C{i}: {sum(st.session_state.clusters==i)}" for i in range(len(np.unique(st.session_state.clusters)))])
                ), unsafe_allow_html=True)
            
            with insight_cols[1]:
                st.markdown("""
                <div class="info-box">
                    <strong>📉 PCA Insights</strong>
                    <ul>
                        <li>Reduced from {} to 2 dimensions</li>
                        <li>Preserves {:.1%} of variance</li>
                        <li>Top PC correlates with: {}
                    </ul>
                </div>
                """.format(
                    st.session_state.X_scaled.shape[1],
                    sum(st.session_state.pca.explained_variance_ratio_),
                    st.session_state.feature_names[np.argmax(abs(st.session_state.pca.components_[0]))]
                ), unsafe_allow_html=True)
            
            with insight_cols[2]:
                st.markdown("""
                <div class="info-box">
                    <strong>💡 Recommendations</strong>
                    <ul>
                        <li>{} clusters is optimal</li>
                        <li>PCA enhances visualization</li>
                        <li>Reduces computational cost
                    </ul>
                </div>
                """.format(
                    len(np.unique(st.session_state.clusters))
                ), unsafe_allow_html=True)

else:
    st.info("👈 Please load a dataset or upload a CSV file from the sidebar to begin!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit, Scikit-learn, and Plotly</p>
    <p>© 2024 Iris Flower Clustering & PCA Visualization</p>
</div>
""", unsafe_allow_html=True)