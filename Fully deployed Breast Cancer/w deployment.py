import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import plotly.express as px
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff
import base64
import time

# Configure the app
st.set_page_config(layout="wide", page_title="Breast Cancer Analysis")
st.title("🧠 Breast Cancer Analysis Dashboard")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.processed = False
    st.session_state.remove_cols = []
    st.session_state.outliers_count = None
    st.session_state.current_tab = "🔍 EDA"  # Track current tab

# Sidebar Navigation - Single Source of Truth
with st.sidebar:
    # New medical/healthcare themed image
    st.image("BC.png",
             width=240,
             caption="Breast Cancer Analysis Dashboard")
    
    st.markdown("## Navigation Menu")
    
    # Main navigation options
    st.session_state.current_tab = st.radio(
        "Select Analysis Phase:",
        ["🔍 Exploratory Data Analysis", 
         "🤖 Machine Learning Modeling", 
         "📈 Advanced Visualizations"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Data Management")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=['csv'])
    
    if uploaded_file:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.original_df = st.session_state.df.copy()
        st.session_state.processed = False  # Reset processing state on new upload
        st.success("Dataset loaded successfully!")
    
    st.markdown("---")
    st.markdown("### About This App")
    st.info("""
    A comprehensive breast cancer analysis tool with:
    - **EDA**: Full data exploration capabilities
    - **ML**: Predictive modeling section
    - **Advanced Viz**: Custom visualization tools
    """)

def create_visualizations(df, title):
    """Create all visualizations using the original plotting code"""
    st.subheader(title)
    
    # 1. Diagnosis Distribution
    st.markdown("### 1. Diagnosis Distribution")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    ax = sns.countplot(x='diagnosis', data=df, hue='diagnosis',
                      palette='muted', legend=False)
    ax.set_title('Count of Diagnosis Categories', fontsize=16, weight='bold')
    ax.set_xlabel('Diagnosis', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}',
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                  ha='center', va='center', fontsize=12,
                  color='black', fontweight='bold', xytext=(0, 10),
                  textcoords='offset points')
    st.pyplot(plt.gcf())
    plt.clf()
    
    # 2. Feature Boxplots
    st.markdown("### 2. Feature Distributions (Boxplots)")
    plt.figure(figsize=(14, 12))
    features = [
        ('radius_mean', 'red'),
        ('texture_mean', 'blue'),
        ('perimeter_mean', 'gray'),
        ('area_mean', 'green'),
        ('compactness_mean', 'lightblue'),
        ('concavity_mean', 'salmon'),
        ('symmetry_mean', 'lightgreen'),
        ('fractal_dimension_mean', 'orange')
    ]
    for i, (feature, color) in enumerate(features):
        plt.subplot(4, 2, i + 1)
        sns.boxplot(x=df[feature], color=color)
        plt.title(f"Boxplot - {feature}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
    
    # 3. Correlation Heatmap
    st.markdown("### 3. Correlation Matrix")
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    plt.figure(figsize=(20, 18))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm',
               fmt=".2f", linewidths=0.5, annot_kws={"size": 10})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.title("Correlation Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
    
    # 4. Histograms by Diagnosis
    st.markdown("### 4. Feature Distributions by Diagnosis")
    columns_to_plot = [
        'radius_mean', 'texture_mean', 'area_mean',
        'perimeter_mean', 'concavity_mean', 'compactness_mean',
        'smoothness_mean'
    ]
    n = len(columns_to_plot)
    plt.figure(figsize=(20, 4 * ((n + 2) // 3)))
    for i, col in enumerate(columns_to_plot):
        plt.subplot((n + 2) // 3, 3, i + 1)
        sns.histplot(data=df, x=col, hue='diagnosis', kde=True,
                    palette='Set2', edgecolor='black', alpha=0.6)
        plt.title(f'Distribution of {col} by Diagnosis')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()
    
    # 5. Pairplot
    st.markdown("### 5. Feature Relationships (Pairplot)")
    selected_cols = [
        'radius_mean', 'texture_mean', 'perimeter_mean',
        'area_mean', 'concavity_mean', 'compactness_mean', 'diagnosis'
    ]
    selected_cols = [col for col in selected_cols if col in df.columns]
    if len(selected_cols) > 1:
        pairplot = sns.pairplot(df[selected_cols], hue='diagnosis', palette={'M': 'red', 'B': 'blue'})
        st.pyplot(pairplot)
    else:
        st.warning("Not enough features available for pairplot")

def detect_outliers(df):
    """Detect outliers using IQR method and return counts per column"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [col for col in numeric_cols if col != 'id']
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound))
    return outliers.sum().to_frame("Outlier Count")

# Main Content Area - Controlled by Sidebar Selection
if st.session_state.df is not None:
    if st.session_state.current_tab == "🔍 Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        
        # EDA Subtabs
        eda_info_tab, eda_process_tab, eda_viz_tab = st.tabs(["📋 Data Information", "⚙️ Data Processing", "📊 Visualizations"])
        
        with eda_info_tab:
            st.header("Dataset Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Basic Info")
                buffer = StringIO()
                st.session_state.df.info(buf=buffer)
                st.text(buffer.getvalue())
                
            with col2:
                st.subheader("Descriptive Statistics")
                st.dataframe(st.session_state.df.describe().style.format("{:.2f}"))
            
            st.subheader("Data Quality Check")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Missing Values**")
                st.dataframe(st.session_state.df.isnull().sum().to_frame("Missing Values"))
            
            with col2:
                st.markdown("**Duplicate Rows**")
                duplicates = st.session_state.df.duplicated().sum()
                st.metric("Total Duplicates", duplicates)
        
        with eda_process_tab:
            st.header("Data Processing Options")
            
            # Column selection
            st.subheader("1. Column Selection")
            st.session_state.remove_cols = st.multiselect(
                "Select columns to remove",
                options=st.session_state.df.columns,
                default=['Unnamed: 32'] if 'Unnamed: 32' in st.session_state.df.columns else None
            )
            
            # Outlier detection
            st.subheader("2. Outlier Detection")
            if st.button("Detect Outliers"):
                st.session_state.outliers_count = detect_outliers(st.session_state.df)
                st.dataframe(st.session_state.outliers_count)
            
            # Outlier handling
            st.subheader("3. Outlier Handling")
            st.markdown("💡 *Capping is generally preferred over removal*")
            outlier_method = st.radio(
                "Select outlier handling method:",
                ["None", "Remove outliers", "Cap outliers"],
                index=2
            )
            
            if st.button("Apply Processing"):
                with st.spinner("Processing data..."):
                    df = st.session_state.original_df.copy()
                    
                    # Remove columns
                    if st.session_state.remove_cols:
                        df = df.drop(st.session_state.remove_cols, axis=1)
                    
                    # Handle outliers
                    if outlier_method != "None":
                        numeric_cols = df.select_dtypes(include=np.number).columns
                        numeric_cols = [col for col in numeric_cols if col != 'id']
                        
                        if outlier_method == "Remove outliers":
                            Q1 = df[numeric_cols].quantile(0.25)
                            Q3 = df[numeric_cols].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            outliers = ((df[numeric_cols] < lower_bound) | (df[numeric_cols] > upper_bound))
                            df = df[~outliers.any(axis=1)]
                        else:
                            for col in numeric_cols:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
                                df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
                    
                    st.session_state.df = df
                    st.session_state.processed = True
                
                st.success("✅ Processing complete! View results in the Visualizations tab.")
        
        with eda_viz_tab:
            if st.session_state.processed:
                before_tab, after_tab = st.tabs(["📉 Before Processing", "📈 After Processing"])

                with before_tab:
                    st.header("Before Handling Outliers")
                    before_df = st.session_state.original_df.copy()
                    if st.session_state.remove_cols:
                        before_df = before_df.drop(st.session_state.remove_cols, axis=1)
                    create_visualizations(before_df, "Original Data")

                with after_tab:
                    st.header("After Handling Outliers")
                    create_visualizations(st.session_state.df, "Processed Data")
            else:
                st.info("Please process your data first in the 'Processing' tab to see visualizations")

    elif st.session_state.current_tab == "🤖 Machine Learning Modeling":

        def prepare_ml_data(df):
            """Prepare data for machine learning"""
            # Move target column to end
            columns = [col for col in df.columns if col != 'diagnosis'] + ['diagnosis']
            ml_data = df[columns]
            
            # Convert diagnosis to numeric (M=1, B=0)
            ml_data['diagnosis'] = ml_data['diagnosis'].map({'M': 1, 'B': 0})
            
            # Drop ID column if exists
            if 'id' in ml_data.columns:
                ml_data = ml_data.drop(columns=['id'])
            
            return ml_data

        def feature_selection(df, threshold=None):
            """Select features based on correlation with target"""
            # Calculate correlations with target
            correlations = df.corr()['diagnosis'].drop('diagnosis')
            
            # Sort by absolute correlation
            correlations_sorted = correlations.abs().sort_values(ascending=False)
            
            # Calculate mean correlation if no threshold provided
            if threshold is None:
                threshold = np.mean(correlations_sorted)
                st.write(f"Using mean correlation as threshold: {threshold:.4f}")
            
            # Select features above threshold
            selected_features = correlations[correlations.abs() > threshold].index.tolist()
            
            # Handle case when no features meet the threshold
            if not selected_features:
                st.warning(f"No features meet the correlation threshold of {threshold}. Using top 3 features instead.")
                selected_features = correlations_sorted.index[:3].tolist()
            
            selected_data = df[selected_features + ['diagnosis']]
            
            return selected_data, selected_features, correlations_sorted

        def train_and_evaluate_models(X_train, X_test, Y_train, Y_test):
            """Train multiple models and evaluate their performance"""
            results = {}
            
            try:
                # Logistic Regression
                lr_model = LogisticRegression(max_iter=10000)
                lr_model.fit(X_train, Y_train)
                lr_pred = lr_model.predict(X_test)
                results['Logistic Regression'] = {
                    'model': lr_model,
                    'accuracy': accuracy_score(Y_test, lr_pred),
                    'f1': f1_score(Y_test, lr_pred),
                    'recall': recall_score(Y_test, lr_pred),
                    'report': classification_report(Y_test, lr_pred, target_names=["Benign", "Malignant"], output_dict=True)
                }
                
                # Decision Tree
                dt_model = DecisionTreeClassifier(random_state=42)
                dt_model.fit(X_train, Y_train)
                dt_pred = dt_model.predict(X_test)
                results['Decision Tree'] = {
                    'model': dt_model,
                    'accuracy': accuracy_score(Y_test, dt_pred),
                    'f1': f1_score(Y_test, dt_pred),
                    'recall': recall_score(Y_test, dt_pred),
                    'report': classification_report(Y_test, dt_pred, target_names=["Benign", "Malignant"], output_dict=True)
                }
                
                # SVM
                svm_model = SVC(kernel='linear', probability=True)
                svm_model.fit(X_train, Y_train)
                svm_pred = svm_model.predict(X_test)
                results['SVM'] = {
                    'model': svm_model,
                    'accuracy': accuracy_score(Y_test, svm_pred),
                    'f1': f1_score(Y_test, svm_pred),
                    'recall': recall_score(Y_test, svm_pred),
                    'report': classification_report(Y_test, svm_pred, target_names=["Benign", "Malignant"], output_dict=True)
                }
                
                # Random Forest
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
                rf_model.fit(X_train, Y_train)
                rf_pred = rf_model.predict(X_test)
                results['Random Forest'] = {
                    'model': rf_model,
                    'accuracy': accuracy_score(Y_test, rf_pred),
                    'f1': f1_score(Y_test, rf_pred),
                    'recall': recall_score(Y_test, rf_pred),
                    'report': classification_report(Y_test, rf_pred, target_names=["Benign", "Malignant"], output_dict=True)
                }
            
            except ValueError as e:
                st.error(f"Model training failed: {str(e)}")
                return None
            st.session_state.X_test = X_test
            st.session_state.Y_test = Y_test

            return results

        def display_model_results(results):
            """Display model evaluation results in Streamlit"""
            if results is None:
                return
                
            for model_name, metrics in results.items():
                st.subheader(model_name)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                with col2:
                    st.metric("F1 Score", f"{metrics['f1']:.4f}")
                with col3:
                    st.metric("Recall", f"{metrics['recall']:.4f}")
                
                st.write("Classification Report:")
                st.dataframe(pd.DataFrame(metrics['report']).transpose())
                st.write("---")

        def predict_new_sample(model, sample_df, feature_columns):
            """Make prediction on new sample and return formatted results"""
            try:
                # Ensure sample has same features as training data
                sample = sample_df[feature_columns]
                prediction = model.predict(sample)
                proba = model.predict_proba(sample)[0]
                
                return {
                    'prediction': 'Malignant (M)' if prediction[0] == 1 else 'Benign (B)',
                    'confidence': f"{max(proba)*100:.1f}%",
                    'malignant_prob': f"{proba[1]*100:.1f}%",
                    'benign_prob': f"{proba[0]*100:.1f}%"
                }
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                return None

        # ML Tab Content
        if st.session_state.df is not None:
            st.header("Machine Learning Modeling")
            
            # Prepare data
            ml_data = prepare_ml_data(st.session_state.df)
            
            # Feature selection section
            st.subheader("1. Feature Selection")
            
            # Dynamic correlation threshold selection
            threshold = st.slider(
                "Correlation threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.4, 
                step=0.01,
                key="corr_threshold",
                help="Features with absolute correlation higher than this value will be selected"
            )
            
            # Get selected features based on current threshold
            selected_data, selected_features, corr_sorted = feature_selection(ml_data, threshold)
            
            # Display correlation table that updates with threshold
            st.markdown("**Selected Features Table**")
            corr_df = pd.DataFrame({
                'Feature': corr_sorted.index,
                'Correlation': corr_sorted.values
            }).set_index('Feature')
            
            # Apply color coding to the table
            def color_selected(val):
                color = 'green' if val else 'red'
                return f'color: {color}'
            
            st.dataframe(
                corr_df.style.format({'Correlation': '{:.4f}'})
                .background_gradient(cmap='coolwarm', subset=['Correlation'])
            )
            
            st.write(f"Number of selected features: {len(selected_features)}")
            st.write("Selected Data Preview:")
            st.dataframe(selected_data.head())
            
            # Model training section
            st.subheader("2. Model Training")
            test_size = st.slider(
                "Test size ratio", 
                min_value=0.1, 
                max_value=0.5, 
                value=0.2, 
                step=0.05,
                key="test_size"
            )
            
            if st.button("Train Models", key="train_models"):
                with st.spinner("Training models..."):
                    # Prepare train/test split
                    X = selected_data.drop(columns='diagnosis', axis=1)
                    Y = selected_data['diagnosis']
                    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42,stratify=Y )
                    
                    # Train and evaluate models
                    results = train_and_evaluate_models(X_train, X_test, Y_train, Y_test)
                    
                    # Store models and selected features in session state
                    if results:
                        st.session_state.models = {name: metrics['model'] for name, metrics in results.items()}
                        st.session_state.selected_features = selected_features
                        st.session_state.X_train = X_train  # Store for getting default values
                        
                        # Display results
                        st.success("Model training completed successfully!")
                        display_model_results(results)
                    else:
                        st.error("Model training failed. Please check your data and parameters.")
            
            # New sample prediction section
            st.subheader("3. Make Predictions")
    
            if 'models' in st.session_state and 'X_test' in st.session_state:

                # Create input form
                with st.form("prediction_form"):
                    st.markdown("**Enter feature values for prediction:**")
                    
                    input_data = {}
                    cols = st.columns(3)
                    
                    # Get default values
                    default_values = {
                        feature: float(st.session_state.X_train[feature].mean())
                        for feature in st.session_state.selected_features
                    }
                    
                    # Create input fields
                    for i, feature in enumerate(st.session_state.selected_features):
                        with cols[i % 3]:
                            input_data[feature] = st.number_input(
                                label=feature,
                                value=default_values[feature],
                                step=0.01,
                                format="%.4f",
                                key=f"input_{feature}"
                            )
                    
                    submitted = st.form_submit_button("Predict Diagnosis")
                    
                    if submitted:
                        sample_df = pd.DataFrame([input_data])
                        
                        st.markdown("**Input Features:**")
                        st.dataframe(sample_df.style.format("{:.4f}"))
                        
                        # Get predictions from all models
                        predictions = {}
                        for name, model in st.session_state.models.items():
                            predictions[name] = predict_new_sample(
                                model, 
                                sample_df, 
                                st.session_state.selected_features
                            )
                        
                        # Display individual model predictions
                        st.markdown("**Model Predictions:**")
                        model_cols = st.columns(len(predictions))
                        
                        for (model_name, pred), col in zip(predictions.items(), model_cols):
                            with col:
                                if pred['prediction'].startswith('Malignant'):
                                    emoji = "🔴"  # Red circle for malignant
                                else:
                                    emoji = "🟢"  # Green circle for benign
                                
                                st.metric(
                                    f"{emoji} {model_name}",
                                    pred['prediction'],
                                    help=f"Confidence: {pred['confidence']}"
                                )
                        
                        # Calculate final weighted decision
                        st.markdown("---")
                        st.subheader("Final Diagnosis Decision")
                        
                        # Define model weights (based on test accuracy)
                        model_weights = {
                            'Logistic Regression': 0.3,
                            'Random Forest': 0.4,  # Highest weight as it's generally most reliable
                            'SVM': 0.2,
                            'Decision Tree': 0.1
                        }
                        
                        # Calculate weighted probabilities
                        malignant_votes = 0
                        benign_votes = 0
                        total_confidence = 0
                        
                        for model_name, pred in predictions.items():
                            weight = model_weights.get(model_name, 0.25)
                            if pred['prediction'].startswith('Malignant'):
                                malignant_votes += weight
                            else:
                                benign_votes += weight
                            total_confidence += float(pred['confidence'].strip('%')) * weight
                        
                        # Determine final prediction
                        final_prediction = "Malignant (M)" if malignant_votes > benign_votes else "Benign (B)"
                        confidence = total_confidence / sum(model_weights.values())
                        certainty = "High" if confidence > 75 else "Medium" if confidence > 60 else "Low"
                        
                        # Display final decision with visual flair
                        if final_prediction == "Malignant (M)":
                            st.error(f"🚨 Final Decision: {final_prediction} (Certainty: {certainty})")
                            st.image("sad.png",
                                    width=260)
                        else:
                            st.success(f"✅ Final Decision: {final_prediction} (Certainty: {certainty})")
                            st.image("happy.png",
                                    width=260)
#____________________________________________________________________________________________                            
                        def analyze_weighted_votes(malignant_votes, benign_votes):
                            total_votes = malignant_votes + benign_votes
                            malignant_percentage = (malignant_votes / total_votes) * 100
                            benign_percentage = (benign_votes / total_votes) * 100
                            
                            if malignant_votes > benign_votes:
                                final_decision = "Malignant (M)"
                                agreement_summary = f"""
                                Malignant received {malignant_percentage:.1f}% of the weighted votes,
                                while Benign received {benign_percentage:.1f}%.
                                Thus, the final diagnosis is **Malignant** based on majority weighted consensus.
                                """
                            else:
                                final_decision = "Benign (B)"
                                agreement_summary = f"""
                                Benign received {benign_percentage:.1f}% of the weighted votes,
                                while Malignant received {malignant_percentage:.1f}%.
                                Thus, the final diagnosis is **Benign** based on majority weighted consensus.
                                """
                            
                            return final_decision, agreement_summary.strip()
    #____________________________________________________________________________________________
                        def generate_medical_report_dynamic(final_decision, certainty, agreement_summary):
                            if final_decision == "Benign (B)":
                                report = f"""
                                -------------------------------
                                🩺 Medical Diagnosis Report
                                -------------------------------

                                Final Decision:  
                                ✅ Benign (B)

                                Agreement Summary:  
                                {agreement_summary}

                                Certainty Level:  
                                {certainty}

                                Recommendation:  
                                Regular monitoring is recommended. Please maintain routine health checks to ensure continued well-being.

                                -------------------------------
                                Doctor's Note  
                                -------------------------------
                                Your health is our priority. If you notice any unusual symptoms, please consult with a specialist for further guidance.
                                """
                            elif final_decision == "Malignant (M)":
                                report = f"""
                                -------------------------------
                                🩺 Medical Diagnosis Report
                                -------------------------------

                                Final Decision:  
                                ❌ Malignant (M)

                                Agreement Summary:  
                                {agreement_summary}

                                Certainty Level:  
                                {certainty}

                                Recommendation:  
                                Immediate consultation with a specialist is strongly advised. Prompt action is essential for effective treatment.

                                -------------------------------
                                Doctor's Note  
                                -------------------------------
                                Your health is critical, and we recommend urgent follow-up. Early intervention is key to successful treatment.
                                """
                            else:
                                report = """
                                -------------------------------
                                ⚠️ Diagnosis Inconclusive
                                -------------------------------

                                Further diagnostic tests are required to confirm the diagnosis.

                                -------------------------------
                                Doctor's Note  
                                -------------------------------
                                Please consult with a medical professional for additional testing.
                                """

                            return report.strip()
    #____________________________________________________________________________________________
                        final_prediction, agreement_summary = analyze_weighted_votes(malignant_votes, benign_votes)
                        medical_report = generate_medical_report_dynamic(final_prediction, certainty, agreement_summary)
    #____________________________________________________________________________________________                           
                        def generate_download_link(report_text, filename="medical_report.txt"):
                            # Encode to base64
                            b64 = base64.b64encode(report_text.encode()).decode()
                            
                            download_link_html = f"""
                            <style>
                            .download-btn-container {{
                                margin: 25px 0;
                                text-align: center;
                            }}
                            
                            .download-btn {{
                                display: inline-flex;
                                align-items: center;
                                justify-content: center;
                                background: linear-gradient(135deg, #3a3a3a, #2a2a2a);
                                color: #f0f0f0;
                                padding: 14px 28px;
                                font-size: 16px;
                                font-weight: 600;
                                text-decoration: none;
                                border-radius: 50px;
                                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
                                transition: all 0.3s ease;
                                border: 1px solid #444;
                                cursor: pointer;
                                position: relative;
                                overflow: hidden;
                            }}
                            
                            .download-btn:hover {{
                                background: linear-gradient(135deg, #444, #333);
                                transform: translateY(-2px);
                                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
                                color: #ffffff;
                                border-color: #555;
                            }}
                            
                            .download-btn:active {{
                                transform: translateY(1px);
                            }}
                            
                            .download-btn::before {{
                                content: "";
                                position: absolute;
                                top: 0;
                                left: -100%;
                                width: 100%;
                                height: 100%;
                                background: linear-gradient(
                                    90deg,
                                    transparent,
                                    rgba(255, 255, 255, 0.1),
                                    transparent
                                );
                                transition: 0.5s;
                            }}
                            
                            .download-btn:hover::before {{
                                left: 100%;
                            }}
                            
                            .download-icon {{
                                margin-right: 10px;
                                font-size: 18px;
                                transition: transform 0.3s ease;
                                filter: brightness(1.2);
                            }}
                            
                            .download-btn:hover .download-icon {{
                                transform: translateY(2px);
                                filter: brightness(1.5);
                            }}
                            </style>
                            
                            <div class="download-btn-container">
                                <a href="data:application/pdf;base64,{b64}" download="{filename}" class="download-btn">
                                    <span class="download-icon">📄</span>
                                    Download Full Medical Report
                                </a>
                            </div>
                            """
                            
                            return download_link_html
                        with st.spinner('🛠️ Generating detailed medical report...'):
                            time.sleep(2)
                            
                            final_prediction, agreement_summary = analyze_weighted_votes(malignant_votes, benign_votes)
                            medical_report = generate_medical_report_dynamic(final_prediction, certainty, agreement_summary)
                            
                            st.subheader("📝 Medical Report")
                            st.info(medical_report)

                            st.markdown(generate_download_link(medical_report), unsafe_allow_html=True)

                        # Show detailed vote breakdown
                        with st.expander("Voting Details"):
                            st.write("**Model Votes:**")
                            vote_df = pd.DataFrame([
                                {
                                    'Model': name,
                                    'Vote': pred['prediction'],
                                    'Weight': f"{model_weights.get(name, 0.25)*100:.0f}%",
                                    'Confidence': pred['confidence']
                                }
                                for name, pred in predictions.items()
                            ])
                            st.dataframe(vote_df)
                            
                            st.write(f"**Weighted Decision:**")
                            st.write(f"- Malignant Votes: {malignant_votes*100:.1f}%")
                            st.write(f"- Benign Votes: {benign_votes*100:.1f}%")
                            st.write(f"- Average Confidence: {confidence:.1f}%")
            else:
                st.info("Please train models first to enable predictions")
        else:
            st.info("Please load and process data in the EDA section first")
    elif st.session_state.current_tab == "📈 Advanced Visualizations":
          # Define consistent color scheme
        COLOR_SCHEME = {
            'B': '#2196F3',  # blue
            'M': '#F44336',  # Red
            'Benign': '#4CAF50',
            'Malignant': '#2196F3',
            '0': '#4CAF50',  # For numeric encoded benign
            '1': '#2196F3'   # For numeric encoded malignant
        }

        # Advanced Visualization Tab
        if st.session_state.df is not None:
            st.header("📊 Advanced Visualizations")
            
            # Create a clean numeric dataframe for visualizations
            viz_df = st.session_state.df.select_dtypes(include=np.number)
            
            # Section 1: Interactive Histogram
            st.subheader("1. Interactive Histogram")
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                hist_feature = st.selectbox(
                    "Select feature for histogram",
                    options=viz_df.columns,
                    index=0
                )
                
                hist_group = st.selectbox(
                    "Group by (histogram)",
                    ["None", "diagnosis"] + st.session_state.df.select_dtypes(exclude=np.number).columns.tolist(),
                    index=0
                )
            
            with hist_col2:
                hist_bins = st.slider("Number of bins", 5, 100, 30)
                hist_height = st.slider("Histogram height", 300, 800, 500)
            
            if hist_group == "None":
                fig_hist = px.histogram(
                    st.session_state.df,
                    x=hist_feature,
                    nbins=hist_bins,
                    height=hist_height,
                    title=f"Distribution of {hist_feature}"
                )
            else:
                fig_hist = px.histogram(
                    st.session_state.df,
                    x=hist_feature,
                    color=hist_group,
                    nbins=hist_bins,
                    color_discrete_map=COLOR_SCHEME,
                    barmode='overlay',
                    opacity=0.7,
                    height=hist_height,
                    title=f"Distribution of {hist_feature} by {hist_group}"
                )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Section 2: Interactive Box Plot
            st.subheader("2. Interactive Box Plot")
            box_col1, box_col2 = st.columns(2)
            
            with box_col1:
                box_feature = st.selectbox(
                    "Select feature for box plot",
                    options=viz_df.columns,
                    index=0
                )
                
                box_by = st.selectbox(
                    "Group by",
                    ["diagnosis"] + st.session_state.df.select_dtypes(exclude=np.number).columns.tolist(),
                    index=0
                )
            
            with box_col2:
                box_log = st.checkbox("Log scale", value=False)
                box_height = st.slider("Box plot height", 300, 800, 500)
            
            fig_box = px.box(
                st.session_state.df,
                x=box_by,
                y=box_feature,
                color=box_by,
                color_discrete_map=COLOR_SCHEME,
                log_y=box_log,
                height=box_height,
                title=f"Distribution of {box_feature} {'by ' + box_by if box_by else ''}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Section 3: Interactive Scatter Plot
            st.subheader("3. Interactive Scatter Plot")
            scatter_col1, scatter_col2 = st.columns(2)
            
            with scatter_col1:
                x_feature = st.selectbox(
                    "X-axis feature",
                    options=viz_df.columns,
                    index=0
                )
                
                y_feature = st.selectbox(
                    "Y-axis feature",
                    options=viz_df.columns,
                    index=1
                )
            
            with scatter_col2:
                color_by = st.selectbox(
                    "Color by",
                    ["diagnosis"] + st.session_state.df.select_dtypes(exclude=np.number).columns.tolist(),
                    index=0
                )
                
                size_by = st.selectbox(
                    "Size by (optional)",
                    [None] + viz_df.columns.tolist(),
                    index=0
                )
            
            fig_scatter = px.scatter(
                st.session_state.df,
                x=x_feature,
                y=y_feature,
                color=color_by,
                color_discrete_map=COLOR_SCHEME,
                size=size_by,
                hover_data=[col for col in st.session_state.df.columns if col not in [x_feature, y_feature]],
                height=600,
                title=f"{y_feature} vs {x_feature}"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Section 4: Network Graph of Feature Correlations
            # Section 4: Network Graph of Feature Correlations
            st.subheader("4. Feature Correlation Network")
            network_col1, network_col2 = st.columns(2)

            with network_col1:
                corr_threshold = st.slider("Correlation threshold", 0.0, 1.0, 0.7, 0.05)
                network_layout = st.selectbox(
                    "Network layout",
                    ['spring', 'circular', 'force', 'random'],
                    index=0
                )

            with network_col2:
                node_size_by = st.selectbox(
                    "Node size by",
                    ['degree', 'importance', 'uniform'],
                    index=0
                )
                show_labels = st.checkbox("Show labels", value=True)

            # Create correlation network
            corr_matrix = viz_df.corr().abs()
            edges = corr_matrix.stack().reset_index()
            edges.columns = ['source', 'target', 'weight']
            edges = edges[edges['weight'] > corr_threshold]
            edges = edges[edges['source'] != edges['target']]

            if not edges.empty:
                G = nx.from_pandas_edgelist(edges, 'source', 'target', 'weight')
                
                # Node sizing
                if node_size_by == 'degree':
                    node_sizes = [d * 500 for n, d in G.degree()]
                elif node_size_by == 'importance' and 'models' in st.session_state:
                    importances = st.session_state.models['Random Forest'].feature_importances_
                    importance_dict = dict(zip(st.session_state.selected_features, importances))
                    node_sizes = [importance_dict.get(n, 0.1) * 2000 for n in G.nodes()]
                else:
                    node_sizes = [30 for n in G.nodes()]
                
                # Create network plot
                pos = nx.spring_layout(G) if network_layout == 'spring' else \
                    nx.circular_layout(G) if network_layout == 'circular' else \
                    nx.random_layout(G)
                
                edge_x = []
                edge_y = []
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')
                
                node_x = []
                node_y = []
                for node in G.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    hoverinfo='text',
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        size=node_sizes,
                        color=node_sizes,
                        colorbar=dict(
                            thickness=15,
                            title='Node Importance',
                            xanchor='left'
                        ),
                        line_width=2))
                
                # Node labels
                node_text = []
                for node in G.nodes():
                    node_text.append(f"{node}<br>Connections: {G.degree()[node]}")
                
                node_trace.text = node_text
                
                fig_network = go.Figure(data=[edge_trace, node_trace],
                                    layout=go.Layout(
                                        title=f'Feature Correlation Network (Threshold: {corr_threshold})',
                                        showlegend=False,
                                        hovermode='closest',
                                        margin=dict(b=20,l=5,r=5,t=40),
                                        height=600))
                
                if show_labels:
                    for node in G.nodes():
                        fig_network.add_annotation(
                            x=pos[node][0], y=pos[node][1],
                            text=node,
                            showarrow=False,
                            font=dict(size=10))
                
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.warning(f"No correlations above {corr_threshold} threshold found.")            
        # In your Model Evaluation section (replace the existing confusion matrix code):

# In your Model Evaluation section:

            if 'models' in st.session_state and 'X_test' in st.session_state and 'Y_test' in st.session_state:
                st.subheader("Model Performance on Test Set")
                
                # Get test data from session state
                X_test = st.session_state.X_test
                Y_test = st.session_state.Y_test
                
                # Create tabs for different views
                tab1, tab2,tab3 = st.tabs(["Confusion Matrices", "Performance Metrics","Detailed Report"])
                
                with tab1:
                    st.markdown("### Test Set Confusion Matrices")
                    
                    # Display one model at a time
                    selected_model = st.selectbox(
                        "Select model to view",
                        options=list(st.session_state.models.keys()),
                        index=0,
                        key="model_select_confusion"
                    )
                    
                    model = st.session_state.models[selected_model]
                    y_pred = model.predict(X_test)
                    cm = confusion_matrix(Y_test, y_pred)
                    
                    # Create clean confusion matrix with just numbers
                    fig = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Benign', 'Malignant'],
                        y=['Benign', 'Malignant'],
                        text_auto=True,
                        color_continuous_scale='Blues',
                        aspect="auto",
                        height=400
                    )
                    
                    fig.update_layout(
                        title=f"{selected_model} - Test Set Performance",
                        xaxis_title="Predicted Label",
                        yaxis_title="True Label",
                        font=dict(size=12)
                    )
                    
                    # Customize hover text
                    fig.update_traces(
                        hovertemplate="<br>".join([
                            "True Label: %{y}",
                            "Predicted Label: %{x}",
                            "Count: %{z}"
                        ])
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add interpretation note
                    st.caption("""
                    **Interpretation**: Rows show true labels, columns show predictions. 
                    Focus on bottom-left (false negatives) as these are dangerous misses.
                    """)
                
                with tab2:
                    st.markdown("### Test Set Metrics Comparison")
                    
                    # Calculate all metrics
                    metrics = []
                    for name, model in st.session_state.models.items():
                        y_pred = model.predict(X_test)
                        probas = model.predict_proba(X_test)[:,1]
                        metrics.append({
                            'Model': name,
                            'Accuracy': accuracy_score(Y_test, y_pred),
                            'Precision': precision_score(Y_test, y_pred),
                            'Recall': recall_score(Y_test, y_pred),
                            'F1 Score': f1_score(Y_test, y_pred),
                            'AUC-ROC': roc_auc_score(Y_test, probas)
                        })
                    
                    metrics_df = pd.DataFrame(metrics).set_index('Model')
                    
                    # Display metrics table
                    st.dataframe(
                        metrics_df.style.format("{:.3f}")
                        .background_gradient(cmap='Blues', subset=['Accuracy', 'F1 Score'])
                        .background_gradient(cmap='Greens', subset=['Recall'])
                        .background_gradient(cmap='Purples', subset=['Precision', 'AUC-ROC'])
                    )
                    
                    # Add metrics explanation
                    with st.expander("Metrics Explanation"):
                        st.markdown("""
                        - **Accuracy**: Overall correct predictions
                        - **Precision**: When predicting malignant, how often correct
                        - **Recall**: What % of malignant cases were caught
                        - **F1**: Balance of precision and recall
                        - **AUC-ROC**: Overall model discrimination ability
                        """)
                with tab3:
                    st.markdown("### Detailed Model Analysis")
                    
                    model_select = st.selectbox(
                        "Select model for detailed report",
                        options=list(st.session_state.models.keys()),
                        index=0,
                        key="model_select_detailed"
                    )
                    
                    model = st.session_state.models[model_select]
                    y_pred = model.predict(X_test)
                    y_proba = model.predict_proba(X_test)[:,1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Classification Report")
                        report = classification_report(
                            Y_test, y_pred,
                            target_names=["Benign", "Malignant"],
                            output_dict=True
                        )
                        st.dataframe(pd.DataFrame(report).transpose().style.format("{:.4f}"))
                    
                    with col2:
                        st.markdown("#### Key Metrics")
                        metrics = {
                            'Accuracy': accuracy_score(Y_test, y_pred),
                            'Precision': precision_score(Y_test, y_pred),
                            'Recall': recall_score(Y_test, y_pred),
                            'F1 Score': f1_score(Y_test, y_pred),
                            'ROC AUC': roc_auc_score(Y_test, y_proba)
                        }
                        st.json(metrics)
                    
                    st.markdown("#### ROC & Precision-Recall Curves")
                    tab_roc, tab_pr = st.tabs(["ROC Curve", "Precision-Recall"])
                    
                    with tab_roc:
                        fpr, tpr, _ = roc_curve(Y_test, y_proba)
                        fig_roc = px.area(
                            x=fpr, y=tpr,
                            title=f'ROC Curve (AUC = {metrics["ROC AUC"]:.4f})',
                            labels=dict(x='False Positive Rate', y='True Positive Rate')
                        )
                        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig_roc, use_container_width=True)
                    
                    with tab_pr:
                        precision, recall, _ = precision_recall_curve(Y_test, y_proba)
                        fig_pr = px.area(
                            x=recall, y=precision,
                            title=f'Precision-Recall Curve (AP = {average_precision_score(Y_test, y_proba):.4f})',
                            labels=dict(x='Recall', y='Precision')
                        )
                        st.plotly_chart(fig_pr, use_container_width=True)

            else:
                st.warning("Please train models first in the ML Modeling tab to see evaluation metrics")              
                
                            # Section 6: Interactive Correlation Heatmap
            st.subheader("6. Interactive Correlation Heatmap")
            heatmap_col1, heatmap_col2 = st.columns(2)
            
            with heatmap_col1:
                corr_method = st.selectbox(
                    "Correlation method",
                    ['pearson', 'kendall', 'spearman'],
                    index=0
                )
                
                annot_toggle = st.checkbox("Show values", value=True)
            
            with heatmap_col2:
                #cluster_toggle = st.checkbox("Cluster features", value=False)
                fig_height = st.slider("Figure height", 500, 1000, 700)
            
            # Calculate correlation matrix
            corr_matrix = viz_df.corr(method=corr_method).fillna(0)
            
            # Create heatmap with red/blue colors (different from our M/B scheme)
            fig_heat = px.imshow(
                corr_matrix,
                labels=dict(x="", y="", color="Correlation"),
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1,
                aspect="auto",
                height=fig_height,
                title="Feature Correlation Matrix"
            )
                        
            if annot_toggle:
                fig_heat.update_traces(text=np.round(corr_matrix.values, 2), 
                                    texttemplate="%{text}")
            
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Section 7: Feature Optimization
            st.subheader("7. Feature Optimization")
            
            if st.checkbox("Show feature importance analysis"):
                if 'models' in st.session_state:
                    rf_model = st.session_state.models['Random Forest']
                    importances = rf_model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    # Create feature importance plot with consistent colors
                    fig_imp = px.bar(
                        x=np.array(st.session_state.selected_features)[indices],
                        y=importances[indices],
                        labels={'x': 'Features', 'y': 'Importance'},
                        title="Random Forest Feature Importance",
                        height=500,
                        color=np.array(st.session_state.selected_features)[indices],
                        color_discrete_sequence=['#2196F3']*len(indices)  # All green for features
                    )
                    fig_imp.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    
                    # Suggest features to keep/drop
                    threshold = st.slider("Importance threshold", 0.0, 0.5, 0.05, 0.01)
                    important_features = [st.session_state.selected_features[i] 
                                        for i in indices if importances[i] > threshold]
                    
                    st.info(f"✅ Suggested features to keep (importance > {threshold}): {', '.join(important_features)}")
                    if len(important_features) < len(st.session_state.selected_features):
                        st.warning(f"❌ Consider dropping: {', '.join([f for f in st.session_state.selected_features if f not in important_features])}")
                else:
                    st.warning("Train models first to see feature importance")
        else:
            st.info("Please load data in the EDA section first")
else:
    st.info("Please upload a Breast Cancer dataset CSV file to begin analysis.")