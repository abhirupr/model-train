import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                             precision_recall_curve, average_precision_score,
                             roc_curve, auc, roc_auc_score, precision_recall_fscore_support)
from sklearn.inspection import permutation_importance
import shap
import xgboost as xgb
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')

class XGBoostClassifier:
    """
    A comprehensive binary classification pipeline that includes:
    - Train/test split
    - Random forest classifier with feature importance analysis (permutation importance and SHAP)
    - Feature selection
    - XGBoost model with hyperparameter tuning via grid search OR hyperopt
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the binary classification model.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        self.rf_model = None
        self.xgb_grid_model = None
        self.xgb_hyperopt_model = None
        self.feature_importance = None
        self.top_features = None
        self.scaler = StandardScaler()
        self.class_imbalance = False
        self.evaluation_metric = 'roc_auc'  # Default, will be updated based on class balance
    
    def load_data(self, X=None, y=None):
        """
        Load and prepare your dataset.
        
        Parameters:
        -----------
        X : DataFrame or array-like, optional
            Feature matrix
        y : Series or array-like, optional
            Target vector
        
        Returns:
        --------
        X : DataFrame
            Feature matrix
        y : Series
            Target vector
        """
        if X is not None and y is not None:
            # Use provided data
            if not isinstance(X, pd.DataFrame):
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=feature_names)
            
            if not isinstance(y, pd.Series):
                y = pd.Series(y, name='target')
            
            return X, y
        
        # If no data provided, create synthetic data for demonstration
        from sklearn.datasets import make_classification
        
        X_synth, y_synth = make_classification(
            n_samples=1000, 
            n_features=20,
            n_informative=10,
            n_redundant=5,
            random_state=self.random_state
        )
        
        feature_names = [f'feature_{i}' for i in range(X_synth.shape[1])]
        X = pd.DataFrame(X_synth, columns=feature_names)
        y = pd.Series(y_synth, name='target')
        
        return X, y
    
    def prepare_data(self, X=None, y=None, test_size=0.2):
        """
        Prepare data for modeling.
        
        Parameters:
        -----------
        X : DataFrame or array-like, optional
            Feature matrix
        y : Series or array-like, optional
            Target vector
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        
        Returns:
        --------
        X_train_scaled : DataFrame
            Scaled training feature matrix
        X_test_scaled : DataFrame
            Scaled test feature matrix
        y_train : Series
            Training target vector
        y_test : Series
            Test target vector
        """
        # Load data if not provided
        if X is None or y is None:
            X, y = self.load_data()
        
        print(f"Dataset shape: {X.shape}")
        class_counts = y.value_counts()
        class_proportions = class_counts / len(y)
        print(f"Class distribution:\n{class_proportions}")
        
        # Check for class imbalance
        min_class_prop = class_proportions.min()
        if min_class_prop < 0.3:  # Arbitrary threshold, you can adjust
            self.class_imbalance = True
            self.evaluation_metric = 'average_precision'
            print("Class imbalance detected. Using PR AUC for evaluation.")
        else:
            self.evaluation_metric = 'roc_auc'
            print("Balanced classes detected. Using ROC AUC for evaluation.")
        
        # Calculate scale_pos_weight for XGBoost
        neg_count = class_counts[0] if 0 in class_counts.index else 0
        pos_count = class_counts[1] if 1 in class_counts.index else 0
        
        if pos_count > 0 and neg_count > 0:
            self.scale_pos_weight = neg_count / pos_count
        elif pos_count == 0:
            raise ValueError("No positive observations found in the dataset. Binary classification requires observations from both classes.")
        elif neg_count == 0:
            raise ValueError("No negative observations found in the dataset. Binary classification requires observations from both classes.")
        
        print(f"scale_pos_weight: {self.scale_pos_weight:.4f}")
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrames to keep feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # Store data for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest(self, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Train Random Forest Classifier and analyze feature importance.
        
        Parameters:
        -----------
        X_train, y_train, X_test, y_test : optional
            If not provided, will use the values stored during prepare_data()
        
        Returns:
        --------
        rf_model : RandomForestClassifier
            Trained Random Forest model
        feature_importance : DataFrame
            Feature importance from different methods
        """
        print("\n" + "="*50)
        print("RANDOM FOREST CLASSIFIER")
        print("="*50)
        
        # Use stored data if not provided
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
            
        feature_names = self.feature_names
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test)
        y_prob = rf_model.predict_proba(X_test)[:, 1]
        
        print("\nRandom Forest Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        if self.class_imbalance:
            pr_auc = average_precision_score(y_test, y_prob)
            print(f"PR AUC: {pr_auc:.4f}")
        else:
            roc_auc = roc_auc_score(y_test, y_prob)
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Class 0', 'Class 1'],
                    yticklabels=['Class 0', 'Class 1'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - Random Forest')
        plt.tight_layout()
        plt.savefig('rf_confusion_matrix.png')
        plt.close()
        
        # Plot ROC curve or PR curve based on class imbalance
        plt.figure(figsize=(8, 6))
        if self.class_imbalance:
            # PR curve for imbalanced datasets
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            plt.plot(recall, precision, lw=2, 
                     label=f'PR curve (AP = {pr_auc:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve - Random Forest')
        else:
            # ROC curve for balanced datasets
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2,
                     label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve - Random Forest')
        
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig('rf_performance_curve.png')
        plt.close()
        
        # 2.1 Feature importance from Random Forest
        plt.figure(figsize=(10, 8))
        importances = pd.Series(rf_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False)
        importances.plot(kind='barh')
        plt.title('Random Forest Built-in Feature Importance')
        plt.tight_layout()
        plt.savefig('rf_feature_importance.png')
        plt.close()
        
        # 2.2 Permutation Importance
        perm_importance = permutation_importance(
            rf_model, X_test, y_test, n_repeats=10, random_state=self.random_state, n_jobs=-1
        )
        
        perm_importances = pd.Series(
            perm_importance.importances_mean, 
            index=feature_names
        ).sort_values(ascending=False)
        
        plt.figure(figsize=(10, 8))
        perm_importances.plot(kind='barh')
        plt.title('Permutation Feature Importance')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        plt.close()
        
        # 2.3 SHAP values
        print("\nCalculating SHAP values for Random Forest (this may take a while)...")
        explainer = shap.TreeExplainer(rf_model)
        # For speed, we use a subset for SHAP calculation
        shap_sample = X_test.sample(min(100, len(X_test)), random_state=self.random_state)
        shap_values = explainer.shap_values(shap_sample)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values, 
                          shap_sample, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        plt.savefig('shap_importance.png')
        plt.close()
        
        # Detailed SHAP summary plot
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values[1] if len(shap_values) > 1 else shap_values, 
                          shap_sample, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        plt.savefig('shap_summary.png')
        plt.close()
        
        # Combine feature importances
        if len(shap_values) > 1:
            shap_importance = np.abs(shap_values[1]).mean(axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
            
        feature_importance = pd.DataFrame({
            'Random Forest': rf_model.feature_importances_,
            'Permutation': perm_importance.importances_mean,
            'SHAP': shap_importance
        }, index=feature_names)
        
        feature_importance = feature_importance.sort_values('SHAP', ascending=False)
        
        # Store model and feature importance
        self.rf_model = rf_model
        self.feature_importance = feature_importance
        
        return rf_model, feature_importance
    
    def select_top_features(self, k=10):
        """
        Select top k features based on feature importance.
        
        Parameters:
        -----------
        k : int, default=10
            Number of top features to select
        
        Returns:
        --------
        top_features : list
            List of selected feature names
        """
        if self.feature_importance is None:
            _, self.feature_importance = self.train_random_forest()
            
        self.top_features = self.feature_importance.index[:k].tolist()
        print(f"\nSelected top {k} features: {self.top_features}")
        
        return self.top_features
        
    def tune_xgboost_with_grid_search(self, k=10, X_train=None, y_train=None, X_test=None, y_test=None):
        """
        Tune XGBoost with GridSearchCV and train on top k features.
        
        Parameters:
        -----------
        k : int, default=10
            Number of top features to select
        X_train, y_train, X_test, y_test : optional
            If not provided, will use the values stored during prepare_data()
        
        Returns:
        --------
        best_grid_model : XGBClassifier
            Best XGBoost model from grid search
        """
        print("\n" + "="*50)
        print(f"XGBOOST WITH GRID SEARCH (TOP {k} FEATURES)")
        print("="*50)
        
        # Use stored data if not provided
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test

        # Select top k features
        if self.top_features is None or len(self.top_features) != k:
            self.select_top_features(k)
    
        X_train_selected = X_train[self.top_features]
        X_test_selected = X_test[self.top_features]
        
        # Create early stopping datasets
        X_train_es, X_valid_es, y_train_es, y_valid_es = train_test_split(
            X_train_selected, y_train, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=y_train
        )
        
        # XGBoost with Grid Search CV - reduced parameter space
        print("\nTraining XGBoost with GridSearchCV and early stopping...")
        
        # Reduced parameter space for practical grid search
        param_grid = {
            'n_estimators': [100, 300, 500, 700, 1000],
            'max_depth': [1, 3, 5, 10, 15, 20, 30],
            'learning_rate': [0.001, 0.01, 0.1, 0.3, 1.0],
            'gamma': [0.0, 0.2, 0.5, 0.8, 1.0],
            'min_child_weight': [1, 5, 10, 15, 20],
            'subsample': [0.1, 0.32, 0.56, 1.0],
            'colsample_bytree': [0.1, 0.32, 0.56, 1.0],
            'colsample_bylevel': [0.1, 0.32, 0.56, 1.0],
            'colsample_bynode': [0.1, 0.32, 0.56, 1.0],
            'scale_pos_weight': np.logspace(np.log10(self.scale_pos_weight), np.log10(self.scale_pos_weight * 10), 4)
        }
        
        print(f"Parameter combinations: {np.prod([len(v) for v in param_grid.values()])}")

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method= 'hist',  # Use histogram-based method for faster training
            grow_policy= 'lossguide',  # For leaf-based tree development
            eval_metric= 'logloss',
            booster= 'gbtree',
            use_label_encoder=False,
            random_state=self.random_state
        )
        
        # We'll use StratifiedKFold for CV to maintain class balance
        stratified_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=stratified_cv,
            scoring=self.evaluation_metric,
            verbose=1,  # Show progress
            n_jobs=-1
        )
        
        # Fit with early stopping
        grid_search.fit(
            X_train_selected, y_train,
            eval_set=[(X_valid_es, y_valid_es)],
            early_stopping_rounds=20,
            verbose=0
        )
        
        print("\nBest parameters from GridSearchCV:")
        print(grid_search.best_params_)
        print(f"Best {self.evaluation_metric} score: {grid_search.best_score_:.4f}")
        
        # Evaluate best model from GridSearchCV
        best_grid_model = grid_search.best_estimator_
        y_grid_pred = best_grid_model.predict(X_test_selected)
        y_grid_prob = best_grid_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nXGBoost GridSearchCV Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_grid_pred):.4f}")
        
        if self.class_imbalance:
            pr_auc = average_precision_score(y_test, y_grid_prob)
            print(f"PR AUC: {pr_auc:.4f}")
        else:
            roc_auc = roc_auc_score(y_test, y_grid_prob)
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_grid_pred))
        
        # Plot performance curve
        self._plot_model_performance(y_test, y_grid_prob, model_name="XGBoost GridSearchCV")
        
        # Feature importance for XGBoost model
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(best_grid_model, max_num_features=k, importance_type='gain', title='XGBoost GridSearchCV Feature Importance')
        plt.tight_layout()
        plt.savefig('xgboost_grid_feature_importance.png')
        plt.close()
        
        # Store the model
        self.xgb_grid_model = best_grid_model
        
        return best_grid_model
    
    def tune_xgboost_with_hyperopt(self, k=10, X_train=None, y_train=None, X_test=None, y_test=None, max_evals=50):
        """
        Tune XGBoost with Hyperopt and train on top k features.
        
        Parameters:
        -----------
        k : int, default=10
            Number of top features to select
        X_train, y_train, X_test, y_test : optional
            If not provided, will use the values stored during prepare_data()
        max_evals : int, default=50
            Maximum number of parameter combinations to try
        
        Returns:
        --------
        final_model : XGBClassifier
            Best XGBoost model from Hyperopt
        """
        print("\n" + "="*50)
        print(f"XGBOOST WITH HYPEROPT (TOP {k} FEATURES)")
        print("="*50)
        
        # Use stored data if not provided
        if X_train is None:
            X_train = self.X_train
        if y_train is None:
            y_train = self.y_train
        if X_test is None:
            X_test = self.X_test
        if y_test is None:
            y_test = self.y_test
    
        # Select top k features
        if self.top_features is None or len(self.top_features) != k:
            self.select_top_features(k)
        
        X_train_selected = X_train[self.top_features]
        X_test_selected = X_test[self.top_features]
        
        # Create early stopping datasets
        X_train_es, X_valid_es, y_train_es, y_valid_es = train_test_split(
            X_train_selected, y_train, 
            test_size=0.2, 
            random_state=self.random_state, 
            stratify=y_train
        )
        
        print("\nTraining XGBoost with Hyperopt and early stopping...")
        
        # Define the search space - exactly matching the parameter ranges
        space = {
            'n_estimators': hp.quniform('n_estimators', 100, 1000, 100),
            'max_depth': hp.quniform('max_depth', 1, 30, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(1)),
            'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001),
            'min_child_weight': hp.quniform('min_child_weight', 1, 20, 1),
            'subsample': hp.loguniform('subsample', np.log(0.1), np.log(1.0)),
            'colsample_bytree': hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)),
            'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0)),
            'colsample_bynode': hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)),
            'scale_pos_weight': hp.loguniform('scale_pos_weight', np.log(self.scale_pos_weight), np.log(self.scale_pos_weight * 10))
        }
        
        def objective(params):
            # Convert hyperopt parameters to proper types
            params = {
                'max_depth': int(params['max_depth']),
                'learning_rate': float(params['learning_rate']),
                'n_estimators': int(params['n_estimators']),
                'gamma': float(params['gamma']),
                'min_child_weight': int(params['min_child_weight']),
                'subsample': float(params['subsample']),
                'colsample_bytree': float(params['colsample_bytree']),
                'colsample_bylevel': float(params['colsample_bylevel']),
                'colsample_bynode': float(params['colsample_bynode']),
                'scale_pos_weight': float(params['scale_pos_weight']),
                'objective': 'binary:logistic',
                'tree_method': 'hist',  # Use histogram-based method for faster training
                'grow_policy': 'lossguide',  # For leaf-based tree development
                'eval_metric': 'logloss',
                'booster': 'gbtree',
                'use_label_encoder': False,
                'random_state': self.random_state
            }
            
            # Create and train XGBoost model
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train_es, y_train_es,
                eval_set=[(X_valid_es, y_valid_es)],
                early_stopping_rounds=20,
                verbose=0
            )
            
            # Predict on validation set
            y_pred = model.predict_proba(X_valid_es)[:, 1]
            
            # Calculate loss based on selected metric
            if self.class_imbalance:
                # For imbalanced classes, use negative PR AUC
                loss = -average_precision_score(y_valid_es, y_pred)
            else:
                # For balanced classes, use negative ROC AUC
                loss = -roc_auc_score(y_valid_es, y_pred)
                
            return {'loss': loss, 'status': STATUS_OK}
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.RandomState(self.random_state)
        )
        
        # Convert best parameters to proper types
        best_params = {
            'max_depth': int(best['max_depth']),
            'learning_rate': best['learning_rate'],
            'n_estimators': int(best['n_estimators']),
            'gamma': best['gamma'],
            'min_child_weight': int(best['min_child_weight']),
            'subsample': best['subsample'],
            'colsample_bytree': best['colsample_bytree'],
            'colsample_bylevel': best['colsample_bylevel'],
            'colsample_bynode': best['colsample_bynode'],
            'scale_pos_weight': best['scale_pos_weight'],
            'objective': 'binary:logistic',
            'use_label_encoder': False,
            'random_state': self.random_state
        }
        
        print("\nBest parameters from Hyperopt:")
        print(best_params)
        
        # Train final model with best parameters
        final_model = xgb.XGBClassifier(**best_params)
        final_model.fit(
            X_train_selected, y_train,
            eval_set=[(X_valid_es, y_valid_es)],
            early_stopping_rounds=20,
            verbose=0
        )
        
        # Evaluate final model
        y_hyperopt_pred = final_model.predict(X_test_selected)
        y_hyperopt_prob = final_model.predict_proba(X_test_selected)[:, 1]
        
        print("\nXGBoost Hyperopt Model Performance:")
        print(f"Accuracy: {accuracy_score(y_test, y_hyperopt_pred):.4f}")
        
        if self.class_imbalance:
            pr_auc = average_precision_score(y_test, y_hyperopt_prob)
            print(f"PR AUC: {pr_auc:.4f}")
        else:
            roc_auc = roc_auc_score(y_test, y_hyperopt_prob)
            print(f"ROC AUC: {roc_auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_hyperopt_pred))
        
        # Plot performance curve
        self._plot_model_performance(y_test, y_hyperopt_prob, model_name="XGBoost Hyperopt")
        
        # Feature importance for final XGBoost model
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(final_model, max_num_features=k, importance_type='gain', title='XGBoost Hyperopt Feature Importance')
        plt.tight_layout()
        plt.savefig('xgboost_hyperopt_feature_importance.png')
        plt.close()
        
        # Store model
        self.xgb_hyperopt_model = final_model
        
        return final_model
    
    def compare_models(self, y_test=None, models=None):
        """
        Compare performance of multiple models on the test set.
        
        Parameters:
        -----------
        y_test : array-like, optional
            Target values for test set
        models : dict, optional
            Dictionary of models to compare {'model_name': (model, predictions)}
        
        Returns:
        --------
        performance : dict
            Dictionary with performance metrics for each model
        """
        if y_test is None:
            y_test = self.y_test
            
        if models is None:
            models = {}
            
            # Add Random Forest if available
            if self.rf_model is not None:
                rf_pred = self.rf_model.predict_proba(self.X_test)[:, 1]
                models['Random Forest'] = (self.rf_model, rf_pred)
                
            # Add XGBoost GridSearch if available
            if self.xgb_grid_model is not None:
                X_test_selected = self.X_test[self.top_features]
                grid_pred = self.xgb_grid_model.predict_proba(X_test_selected)[:, 1]
                models['XGBoost GridSearch'] = (self.xgb_grid_model, grid_pred)
                
            # Add XGBoost Hyperopt if available
            if self.xgb_hyperopt_model is not None:
                X_test_selected = self.X_test[self.top_features]
                hyperopt_pred = self.xgb_hyperopt_model.predict_proba(X_test_selected)[:, 1]
                models['XGBoost Hyperopt'] = (self.xgb_hyperopt_model, hyperopt_pred)
        
        if not models:
            print("No models available for comparison. Train models first.")
            return None
            
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Compare models
        performance = {}
        plt.figure(figsize=(10, 8))
        
        for name, (model, y_prob) in models.items():
            # Calculate metrics
            y_pred = (y_prob >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            
            if self.class_imbalance:
                # PR AUC for imbalanced datasets
                pr_auc = average_precision_score(y_test, y_prob)
                performance[name] = {'accuracy': accuracy, 'pr_auc': pr_auc}
                
                # Plot PR curve
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP = {pr_auc:.4f})')
            else:
                # ROC AUC for balanced datasets
                roc_auc = roc_auc_score(y_test, y_prob)
                performance[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}
                
                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
        
        # Set plot labels based on class balance
        if self.class_imbalance:
            baseline = np.sum(y_test) / len(y_test)
            plt.axhline(y=baseline, linestyle='--', color='r', label=f'Baseline (AP = {baseline:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve Comparison')
        else:
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve Comparison')
        
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Print performance metrics
        print("\nPerformance Summary:")
        for name, metrics in performance.items():
            print(f"\n{name}:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            if 'roc_auc' in metrics:
                print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
            if 'pr_auc' in metrics:
                print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        
        return performance
    
    def _plot_model_performance(self, y_test, y_prob, model_name="Model"):
        """
        Plot performance curve for a model.
        
        Parameters:
        -----------
        y_test : array-like
            True target values
        y_prob : array-like
            Predicted probabilities
        model_name : str, default="Model"
            Name of the model for plot titles
        """
        plt.figure(figsize=(8, 6))
        
        if self.class_imbalance:
            # PR curve for imbalanced datasets
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
            
            plt.plot(recall, precision, lw=2, label=f'PR curve (AP = {pr_auc:.4f})')
            baseline = np.sum(y_test) / len(y_test)
            plt.axhline(y=baseline, linestyle='--', color='r', label=f'Baseline (AP = {baseline:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
        else:
            # ROC curve for balanced datasets
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
        
        plt.legend(loc="best")
        plt.tight_layout()
        
        # Create safe filename from model name
        safe_name = model_name.lower().replace(' ', '_')
        plt.savefig(f'{safe_name}_performance_curve.png')
        plt.close()
    
    def fit(self, X=None, y=None, test_size=0.2, n_features=10, tuning_method='both'):
        """
        Complete pipeline: prepare data, train Random Forest, 
        analyze feature importance, and train XGBoost models.
        
        Parameters:
        -----------
        X : DataFrame or array-like, optional
            Feature matrix
        y : Series or array-like, optional
            Target vector
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        n_features : int, default=10
            Number of top features to select for XGBoost models
        tuning_method : str, default='both'
            Which tuning method to use ('grid', 'hyperopt', or 'both')
            
        Returns:
        --------
        self : BinaryClassificationModel
            The fitted model instance
        """
        print("BINARY CLASSIFICATION PIPELINE")
        print("=" * 50)
        
        # 1. Prepare data
        self.prepare_data(X, y, test_size)
        
        # 2. Train Random Forest and analyze feature importance
        self.train_random_forest()
        
        # Print top features from different methods
        print("\nTop 10 features by different importance methods:")
        print(self.feature_importance[['Random Forest', 'Permutation', 'SHAP']].head(10))
        
        # 3. Select top features
        self.select_top_features(k=n_features)
        
        # 4. Train XGBoost with selected tuning method
        if tuning_method.lower() == 'grid' or tuning_method.lower() == 'both':
            self.tune_xgboost_with_grid_search(k=n_features)
        
        if tuning_method.lower() == 'hyperopt' or tuning_method.lower() == 'both':
            self.tune_xgboost_with_hyperopt(k=n_features)
            
        # 5. Compare models
        if tuning_method.lower() == 'both':
            self.compare_models()
        
        print("\nBinary Classification Pipeline Completed!")
        return self
    
    def predict(self, X, model='hyperopt'):
        """
        Make predictions with the selected model.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix to make predictions on
        model : str, default='hyperopt'
            Which model to use for predictions ('hyperopt', 'grid', 'rf')
        
        Returns:
        --------
        y_pred : array
            Predicted class labels
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        if model.lower() == 'hyperopt' and self.xgb_hyperopt_model is not None:
            # Select features and predict with hyperopt model
            X_selected = X_scaled[self.top_features]
            return self.xgb_hyperopt_model.predict(X_selected)
        elif model.lower() == 'grid' and self.xgb_grid_model is not None:
            # Select features and predict with grid search model
            X_selected = X_scaled[self.top_features]
            return self.xgb_grid_model.predict(X_selected)
        elif model.lower() == 'rf' and self.rf_model is not None:
            # Predict with random forest model
            return self.rf_model.predict(X_scaled)
        else:
            raise ValueError(f"Model '{model}' not available or not fitted yet")
    
    def predict_proba(self, X, model='hyperopt'):
        """
        Predict class probabilities with the selected model.
        
        Parameters:
        -----------
        X : DataFrame or array-like
            Feature matrix to make predictions on
        model : str, default='hyperopt'
            Which model to use for predictions ('hyperopt', 'grid', 'rf')
        
        Returns:
        --------
        y_prob : array
            Predicted class probabilities
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        if model.lower() == 'hyperopt' and self.xgb_hyperopt_model is not None:
            # Select features and predict with hyperopt model
            X_selected = X_scaled[self.top_features]
            return self.xgb_hyperopt_model.predict_proba(X_selected)
        elif model.lower() == 'grid' and self.xgb_grid_model is not None:
            # Select features and predict with grid search model
            X_selected = X_scaled[self.top_features]
            return self.xgb_grid_model.predict_proba(X_selected)
        elif model.lower() == 'rf' and self.rf_model is not None:
            # Predict with random forest model
            return self.rf_model.predict_proba(X_scaled)
        else:
            raise ValueError(f"Model '{model}' not available or not fitted yet")
    
    def get_feature_importance(self):
        """
        Get feature importance data from the models.
        
        Returns:
        --------
        feature_importance : DataFrame
            Feature importance from different methods
        """
        if self.feature_importance is None:
            raise ValueError("Models have not been trained yet. Call fit() first.")
        return self.feature_importance