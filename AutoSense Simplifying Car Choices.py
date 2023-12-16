import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import graphviz
from scipy.sparse import hstack
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from PIL import Image

st.set_page_config(
    page_title="Custom Streamlit Styling",
    page_icon=":smiley:",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.set_option('deprecation.showPyplotGlobalUse', False)

def Preprocessing(dataset):
    number_mapping = {
    '2': 'two',
    '3': 'three',
    '4': 'four'
    }

    column_name = 'persons'
    dataset[column_name] = dataset[column_name].astype(str)
    dataset[column_name] = dataset[column_name].replace(number_mapping)
    dataset.to_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv', index=False)
    
    number_mapping2 = {
    '2': 'two',
    '3': 'three',
    '4': 'four',
}

    column_name = 'doors'

    dataset[column_name] = dataset[column_name].astype(str)
    dataset[column_name] = dataset[column_name].replace(number_mapping)
    dataset.to_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv', index=False)

    st.write("Preprocessed data:")
    for col in dataset.columns:
        if dataset[col].dtype == 'object' and dataset[col].isna().any():
            # Impute missing values with the mode (most frequent category)
            mode_val = dataset[col].mode()[0]
            dataset[col].fillna(mode_val, inplace=True)

    print("\nDataset after handling missing values:")
    return dataset


def home():
    st.title("Car Dataset Analysis using Decision tree")
    st.write("By 21PD22 - Nilavini")
    st.write("   21PD27 - Raja Neha")
    st.write("   21PD39 - Varsha")
    st.header("About the dataset")
    
    with open("style.css", "r") as f:
            css = f.read()
            st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

            #st.markdown("""<div class="boxed-paragraph">Participants were 61 children with ADHD and 60 healthy controls (boys and girls, ages 7-12). EEG recording was performed based on 10-20 standard by 19 channels (Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2) at 128 Hz sampling frequency. The A1 and A2 electrodes were the references located on earlobes.Since one of the deficits in ADHD children is visual attention, the EEG recording protocol was based on visual attention tasks.    In the task, a set of pictures of cartoon characters was shown to the children and they were asked to count the characters. The number of characters in each image was randomly selected between 5 and 16, and the size of the pictures was large enough to be easily visible and countable by children. To have a continuous stimulus during the signal recording, each image was displayed immediately and uninterrupted after the child's response.  Thus, the duration of EEG recording throughout this cognitive visual task was dependent on the child's performance (i.e. response speed)""", unsafe_allow_html=True)
            st.markdown("""Number of Instances: 1728""")
            st.write("Number of Attributes: 6")
            st.write("\n")
            st.markdown("Attribute Values: ")
            st.markdown("buying price: v-high, high, med, low")
            st.markdown("maintanence cost: v-high, high, med, low")
            st.markdown("number of doors: 2, 3, 4, 5-more")
            st.markdown("number of persons: 2, 4, more")
            st.markdown("lug_boot: small, med, big")
            st.markdown("safety: low, med, high")
            st.markdown("decision: unacc,acc,good,vgood")
            
            dataset = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car.csv')
            st.write("\nDataset:")
            st.write(dataset)
            
            dataset = Preprocessing(dataset)
            st.write(dataset)
    
def Model():
        st.markdown("Decsion tree")
        dataset = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')
        label_encoder = LabelEncoder()

        # Applying Label Encoding to your categorical columns
        categorical_cols = ['buying cost', 'maintainence cost', 'doors', 'persons', 'lug_boot', 'safety']
        for col in categorical_cols:
            dataset[col] = label_encoder.fit_transform(dataset[col])
        
        # Applying One-Hot Encoding to your categorical columns
        data = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)

        X = data.drop('decision', axis=1)  # Features
        y = data['decision']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
        dt_classifier.fit(X_train, y_train)
        
        y_pred = dt_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy:", accuracy)

        classification_rep = classification_report(y_test, y_pred)
        st.text(f'Classification Report:\n{classification_rep}\n')

        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write(f'Confusion Matrix:\n{conf_matrix}')
        
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Create a heatmap of the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", linewidths=.5)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        st.pyplot()
        

        plt.figure(figsize=(20, 10))

        plot_tree(dt_classifier, filled=True, feature_names=X.columns, class_names=dt_classifier.classes_)
        plt.title('Decision Tree')

        # Saving the plot to an image file (e.g., 'decision_tree.png')
        plt.savefig('decision_tree.png', format='png', bbox_inches='tight')

        image = Image.open('decision_tree.png')

        # Display the image using st.image()
        st.image(image, caption='Decision tree', use_column_width=True)

        
        """
        dot_data = export_graphviz(
        dt_classifier,
        out_file=None,
        feature_names=list(X.columns),
        class_names=dt_classifier.classes_,
        filled=True,
        rounded=True,
        special_characters=True
        )
        graph = graphviz.Source(dot_data)
        graph.render('decision_tree', view=True)
        
        with open("decision_tree.pdf", "rb") as pdf_file:
                PDFbyte = pdf_file.read()

        st.download_button(label="Export_Report",
                        data=PDFbyte,
                        file_name="decisiontree.pdf",
                        mime='application/octet-stream')

        """
def plots_and_EDA():

        st.markdown("Plots")
        dataset = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')
        sns.countplot(x='decision', data=dataset)
        plt.title("Class Distribution of 'Decision'")
        plt.show()
        st.pyplot()     
        
        plt.figure(figsize=(12, 5))

        # Count plot for "doors"
        plt.subplot(1, 2, 1)
        sns.countplot(x='doors', data=dataset, hue='decision')
        plt.title("Decision based on number of doors")
        # Count plot for "persons"
        plt.subplot(1, 2, 2)
        sns.countplot(x='persons', data=dataset, hue='decision')
        plt.title("Decision based on number of Persons")

        plt.tight_layout()
        plt.show()
        st.pyplot()
        
        categorical_features = ["buying cost", "maintainence cost", "doors", "persons", "lug_boot", "safety"]
        for feature in categorical_features:
            plt.figure(figsize=(10, 5))
            sns.countplot(x=feature, data=dataset, hue='decision')
            plt.title(f"Count of {feature} by 'decision'")
            plt.xticks(rotation=45)
            plt.show()
            st.pyplot()
            
        for col in dataset.columns:
            if dataset[col].dtype == 'object':
                value_counts = dataset[col].value_counts()
                sns.countplot(x=col, data=dataset)
                plt.title(f'Countplot of {col}')
                plt.show()
                st.pyplot()



        
def K_Fold():
        dataset = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')
        label_encoder = LabelEncoder()

        # Applying Label Encoding to your categorical columns
        categorical_cols = ['buying cost', 'maintainence cost', 'doors', 'persons', 'lug_boot', 'safety']
        for col in categorical_cols:
            dataset[col] = label_encoder.fit_transform(dataset[col])
        data = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)

        X = data.drop('decision', axis=1)  # Features
        y = data['decision']  # Target variable
        def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
                plt.figure()
                plt.title(title)
                if ylim is not None:
                        plt.ylim(*ylim)
                plt.xlabel("Training examples")
                plt.ylabel("Score")
                train_sizes, train_scores, test_scores = learning_curve(
                        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                plt.grid()

                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                train_scores_mean + train_scores_std, alpha=0.1,
                                color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                        label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                        label="Cross-validation score")

                plt.legend(loc="best")
                return plt

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Decision Tree classifier
        estimator = DecisionTreeClassifier()

        # Plot learning curves
        plot_learning_curve(estimator, "Decision Tree Learning Curve", X_train, y_train, cv=5, n_jobs=-1)

        plt.show()
        st.pyplot()
        
        clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
        
        kf = KFold(n_splits=5, random_state=42, shuffle=True)



        for i, (train_index, val_index) in enumerate(kf.split(X, y)):
                X_train, X_val = X.iloc[train_index], X.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                clf.fit(X_train, y_train)

                # Calculate training accuracy for this fold
                y_train_pred = clf.predict(X_train)
                training_accuracy = accuracy_score(y_train, y_train_pred)

                # Calculate validation accuracy for this fold
                y_val_pred = clf.predict(X_val)
                validation_accuracy = accuracy_score(y_val, y_val_pred)

                st.write(f"Fold {i + 1}: Training Accuracy = {training_accuracy:.2f}, Validation Accuracy = {validation_accuracy:.2f}")
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
                dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
                dt_classifier.fit(X_train, y_train)

                # Train your model on the training set (you may have already done this)
                clf.fit(X_train, y_train)

                # Evaluate the model on the test set
                y_pred_test = clf.predict(X_test)

                # Calculate the test accuracy
                test_accuracy = accuracy_score(y_test, y_pred_test)
                st.write(f'Test Accuracy: {test_accuracy:.2f}')
                
        param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
        }

        # Create a GridSearchCV object
        grid_search = GridSearchCV(
                estimator= dt_classifier,
                param_grid=param_grid,
                cv=5,  # 5-fold cross-validation
                n_jobs=-1,  # Use all available CPU cores for parallelization
                verbose=1
        )

        # Fit the model to the data while searching for the best hyperparameters
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters
        best_params = grid_search.best_params_
        
        best_dt_classifier = DecisionTreeClassifier(
        criterion=grid_search.best_params_['criterion'],
        max_depth=grid_search.best_params_['max_depth'],
        min_samples_split=grid_search.best_params_['min_samples_split'],
        min_samples_leaf=grid_search.best_params_['min_samples_leaf']
        )

        # Fit the model with the training data
        best_dt_classifier.fit(X_train, y_train)

        # Evaluate the model on the test data
        y_pred = best_dt_classifier.predict(X_test)

        # Calculate accuracy and other evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Print or visualize the evaluation results
        st.write("Accuracy:", accuracy)
        st.text("Classification Report:\n" + classification_rep)
        st.write(f'Confusion Matrix:\n{conf_matrix}')

        

def RandomForest():
        encoder = OneHotEncoder()
        label_encoder = LabelEncoder()

        categorical_cols = ['buying cost', 'maintainence cost', 'doors', 'persons', 'lug_boot', 'safety']
        
        dataset = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')
        
        X = dataset.drop('decision', axis=1)  # Features
        y = dataset['decision']  # Target variable
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Apply One-Hot Encoding to the categorical features
        X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
        X_test_encoded = encoder.transform(X_test[categorical_cols])

        # Now, you can concatenate the one-hot encoded features with the other numeric features

        X_train_encoded = hstack((X_train_encoded, X_train.drop(categorical_cols, axis=1)))
        X_test_encoded = hstack((X_test_encoded, X_test.drop(categorical_cols, axis=1)))

        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        # Fit the Random Forest classifier to the training data
        rf_classifier.fit(X_train_encoded, y_train)

        # Make predictions on the test data
        y_pred_rf = rf_classifier.predict(X_test_encoded)
        
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f'Random Forest Accuracy: {accuracy_rf:.2f}\n')

        # Print a classification report
        classification_rep_rf = classification_report(y_test, y_pred_rf)
        st.text(f'Random Forest Classification Report:\n{classification_rep_rf}\n')

        # Create a confusion matrix
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        st.write(f'Random Forest Confusion Matrix:\n{conf_matrix_rf}')
        
def ID3(): 
        df = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')
        X = df.drop('decision', axis=1)  # Features
        y = df['decision']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        test_data=X_test
        test_data['decision']=y_test
        train_data=X_train
        train_data['decision']=y_train
        
        totalNoRows=train_data.shape[0]
        classLabels=train_data.decision.unique()
        #print("Total number of classes : ", len(classLabels))
        countClasses= len(classLabels)
        total_entropy=0
        for i in classLabels:
                class_count = train_data[train_data["decision"] == i].shape[0]
                class_entropy = - (class_count/totalNoRows)*np.log2(class_count/totalNoRows)
                total_entropy = total_entropy+class_entropy
        st.write("Entropy of the dataset : ",total_entropy)
        
        def entropy_of_Features(feature,col):
                l=feature.shape[0]
                total=0
                groups =feature[col].groupby(feature['decision'])
                for name,group in groups:
                        count=group.shape[0]
                        entropy=-(count/l)*np.log2(count/l)
                        #print(name,entropy)
                        total=total+entropy
                #print(total)
                return total
        values=[]
        categorical_features = ["buying cost", "maintainence cost", "doors", "persons", "lug_boot", "safety"]
        for i in categorical_features:
                #print("\n",i,"\n")
                groups=df.groupby(df[i])
                lt=[]
                for name,group in groups:
                        #print(name)
                        lt.append(entropy_of_Features(group,i))
                values.append(lt)
        
        def information_gain(feature,df,classLabels):
                feature_values=df[feature].unique()
                info=0.0
                for i in feature_values:
                        value_data = df[df[feature] ==i] #filtering rows with that feature_value
                        value_count = value_data.shape[0]
                        feature_entropy=entropy_of_Features(value_data,feature)
                        probability = value_count/totalNoRows
                        info +=  probability * feature_entropy
                return  total_entropy-info

        information_gain("buying cost",train_data,classLabels)
        
        def highest_informative_feature(df,classLabels):
                dataX=df.columns.drop('decision')
                maxInfoGain=-1
                featureName=""
                for i in categorical_features:
                        infoGain=information_gain(i,df,classLabels)
                        print(i,infoGain)
                        if infoGain>maxInfoGain:
                                maxInfoGain=infoGain
                                featureName=i
                return featureName
        
        def id3(df, originaldata, features, classLabels):
                if len(np.unique(df['decision'])) <= 1:
                        return np.unique(df['decision'])[0]
                elif len(df) == 0:
                        return np.unique(originaldata['decision'])[np.argmax(np.unique(originaldata['decision'], return_counts=True)[1])]
                elif len(features) == 0:
                        return np.unique(df['decision'])[np.argmax(np.unique(df['decision'], return_counts=True)[1])]
                else:
                        best_feature = highest_informative_feature(df, classLabels)
                        Dtree = {best_feature: {}}
                        features = [i for i in features if i != best_feature]
                        for value in np.unique(df[best_feature]):
                                value_data = df.where(df[best_feature] == value).dropna()
                                subtree = id3(value_data, df, features, classLabels)
                                Dtree[best_feature][value] = subtree
                        return Dtree


        Dtree = id3(train_data,train_data, categorical_features, classLabels)
        
        def visualize_custom_tree(tree, feature_names, class_names):
                def display_tree(node, depth=0):
                        if isinstance(node, dict):
                                for feature, subtree in node.items():
                                        st.markdown('  ' * depth + feature + ":")
                                        display_tree(subtree, depth + 1)
                                else:
                                        st.markdown('  ' * depth + "Class " + str(node))

                display_tree(tree)

        def predict(query, tree, default=1):
                for key in list(query.keys()):
                        if key in list(tree.keys()):
                                try:
                                        result = tree[key][query[key]]
                                except:
                                        return default
                                result = tree[key][query[key]]
                                if isinstance(result, dict):
                                        return predict(query, result)
                                else:
                                        return result
        y_pred=[]
        for index, row in test_data.iterrows():
                query = row.drop('decision').to_dict()  # Convert the row to a dictionary query
                prediction = predict(query, Dtree)  # Apply the prediction function to the query
                #print(f"Prediction for row {index}: {prediction}")
                y_pred.append(prediction)
        def evaluate_model(y_true, y_pred):
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, average='weighted')
                recall = recall_score(y_true, y_pred, average='weighted')
                f1 = f1_score(y_true, y_pred, average='weighted')

                st.write(f"Accuracy: {accuracy:.2f}")
                st.write(f"Precision: {precision:.2f}")
                st.write(f"Recall: {recall:.2f}")

        evaluate_model(y_test,y_pred)

        # Replace 'feature_names' and 'class_names' with the appropriate variables from your dataset
        feature_names = df.columns.drop('decision').tolist()
        class_names = np.unique(df['decision']).tolist()

        # Visualize the decision tree
        visualize_custom_tree(Dtree, feature_names, class_names)


def pate_algo():
        st.markdown("PATE Algorithm")
        
        data = pd.read_csv('C:\\Users\\birth\\ML sem hackathon\\car1.csv')

        # Perform label encoding for categorical columns
        label_encoder = LabelEncoder()
        categorical_columns = ['buying cost', 'maintainence cost', 'doors', 'persons', 'lug_boot', 'safety', 'decision']

        for column in categorical_columns:
                data[column] = label_encoder.fit_transform(data[column])
        
        X = data.drop(columns=['decision'])
        y = data['decision']
        X_public, X_private, y_public, y_private = train_test_split(X, y, test_size=0.5, random_state=42)
        
        teacher_bob = DecisionTreeClassifier(max_depth=5)
        teacher_bob.fit(X_public, y_public)

        teacher_alice = DecisionTreeClassifier(max_depth=5)
        teacher_alice.fit(X_public, y_public)

        predictions_bob = teacher_bob.predict(X_private)
        predictions_alice = teacher_alice.predict(X_private)

        # Add Laplace noise to teacher predictions
        epsilon = 1.0  # Adjust for desired privacy level
        noisy_predictions_bob = predictions_bob + np.random.laplace(scale=1.0 / epsilon, size=len(predictions_bob))
        noisy_predictions_alice = predictions_alice + np.random.laplace(scale=1.0 / epsilon, size=len(predictions_alice))

        
        student_model = DecisionTreeClassifier(max_depth=5)
        student_model.fit(np.vstack([noisy_predictions_bob, noisy_predictions_alice]).T, y_public)

        # Generate predictions from the student model
        student_predictions = student_model.predict(np.vstack([noisy_predictions_bob, noisy_predictions_alice]).T)

        # Evaluate the student model's accuracy
        accuracy = accuracy_score(y_public, student_predictions)
        st.write("Student Model Accuracy: {:.2f}%".format(accuracy * 100))


def main():

        st.sidebar.title("Analysis")
        page = st.sidebar.selectbox("Select a Page", ["About the Dataset", "Model", "plots","K Fold","Random Forest","ID3","PATE"])
            
        if page == "About the Dataset":
                home()
        elif page == "Model":
                Model()
        elif page == "plots":
                plots_and_EDA()
        elif page == "K Fold":
                K_Fold()
        elif page == "Random Forest":
                RandomForest()
        elif page == "ID3":
                ID3()
        elif page == "PATE":
                pate_algo()
                
        

if __name__ == "__main__":
    
    main()