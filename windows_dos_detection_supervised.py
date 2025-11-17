#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoS Detection using Supervised Learning Methods
Uses labeled CICIDS2017 dataset for training

Methods: XGBoost, Random Forest, LightGBM
Comparison with Isolation Forest (unsupervised)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARNING] XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARNING] LightGBM not available. Install with: pip install lightgbm")

class SupervisedDoSDetector:
    """
    Supervised Learning based DoS Detector
    
    Uses labeled data (both normal and DoS) for training
    Methods: XGBoost, Random Forest, LightGBM
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.label_encoder = LabelEncoder()
        
    def load_and_prepare(self, normal_file=None, attack_files=None, sample_size=30000):
        """
        Load CICIDS2017 and prepare for supervised learning
        """
        print("="*70)
        print("LOADING CICIDS2017 DATASET (SUPERVISED LEARNING)")
        print("="*70)
        
        # Load normal traffic
        df_normal = pd.DataFrame()
        label_col = None
        
        if normal_file and os.path.exists(normal_file):
            print(f"\nNormal traffic file: {os.path.basename(normal_file)}")
            df_normal_full = pd.read_csv(normal_file, encoding='latin1')
            df_normal_full.columns = df_normal_full.columns.str.strip()
            
            for col in df_normal_full.columns:
                if 'label' in col.lower():
                    label_col = col
                    break
            
            df_normal = df_normal_full[df_normal_full[label_col].str.strip() == 'BENIGN'].copy()
            print(f"  Normal flows: {len(df_normal):,}")
        
        # Load attack traffic from multiple files
        df_dos_list = []
        df_attack_full_list = []
        
        if attack_files is None:
            attack_files = []
        if not isinstance(attack_files, list):
            attack_files = [attack_files] if attack_files else []
        
        for attack_file in attack_files:
            if attack_file and os.path.exists(attack_file):
                print(f"\nAttack traffic file: {os.path.basename(attack_file)}")
                df_attack_full = pd.read_csv(attack_file, encoding='latin1')
                df_attack_full.columns = df_attack_full.columns.str.strip()
                df_attack_full_list.append(df_attack_full)
                
                if label_col is None:
                    for col in df_attack_full.columns:
                        if 'label' in col.lower():
                            label_col = col
                            break
                
                # Get DoS/DDoS attacks only
                dos_keywords = ['DoS', 'DDoS', 'dos', 'ddos']
                df_dos_file = df_attack_full[df_attack_full[label_col].str.contains('|'.join(dos_keywords), case=False, na=False)].copy()
                df_dos_list.append(df_dos_file)
                print(f"  DoS/DDoS flows from this file: {len(df_dos_file):,}")
                
                if len(df_normal) == 0:
                    df_normal_file = df_attack_full[df_attack_full[label_col].str.strip() == 'BENIGN'].copy()
                    if len(df_normal) == 0:
                        df_normal = df_normal_file
                    else:
                        df_normal = pd.concat([df_normal, df_normal_file], ignore_index=True)
        
        # Combine all DoS/DDoS flows
        if df_dos_list:
            df_dos = pd.concat(df_dos_list, ignore_index=True)
            print(f"\n  Total DoS/DDoS flows (combined): {len(df_dos):,}")
        else:
            df_dos = pd.DataFrame()
        
        if label_col is None:
            raise ValueError("Label column not found")
        
        print(f"\nLabel column: '{label_col}'")
        print(f"\nTraffic breakdown:")
        print(f"  Normal (BENIGN):  {len(df_normal):,}")
        print(f"  DoS/DDoS attacks: {len(df_dos):,}")
        
        if len(df_normal) == 0 or len(df_dos) == 0:
            raise ValueError("Insufficient data: Need both normal and DoS traffic")
        
        # Sample for faster processing
        if len(df_normal) > sample_size:
            df_normal = df_normal.sample(n=sample_size, random_state=42)
        if len(df_dos) > sample_size // 3:
            df_dos = df_dos.sample(n=sample_size // 3, random_state=42)
        
        print(f"\nSampled data:")
        print(f"  Normal:  {len(df_normal):,}")
        print(f"  DoS:     {len(df_dos):,}")
        print(f"  Total:   {len(df_normal) + len(df_dos):,} samples")
        
        # Combine for feature extraction
        if df_attack_full_list:
            df_for_features = pd.concat(df_attack_full_list, ignore_index=True)
        elif normal_file and os.path.exists(normal_file):
            df_for_features = df_normal_full
        else:
            df_for_features = pd.concat([df_normal, df_dos], ignore_index=True)
        
        # Get all numeric features
        numeric_cols = df_for_features.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = [label_col, 'Destination Port']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\nUsing {len(feature_cols)} numeric features")
        
        # Prepare data
        df_normal['Label'] = 0  # 0 = Normal
        df_dos['Label'] = 1     # 1 = DoS
        
        # Combine
        df_combined = pd.concat([df_normal, df_dos], ignore_index=True)
        
        # Extract features and labels
        X = df_combined[feature_cols].copy()
        y = df_combined['Label'].copy()
        
        # Clean data
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.feature_cols = feature_cols
        
        return X, y
    
    def train_models(self, X_train, y_train):
        """
        Train multiple supervised models
        """
        print("\n" + "="*70)
        print("TRAINING SUPERVISED MODELS")
        print("="*70)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nTraining samples: {len(X_train):,}")
        print(f"Features: {len(self.feature_cols)}")
        print(f"Normal samples: {(y_train == 0).sum():,}")
        print(f"DoS samples: {(y_train == 1).sum():,}")
        
        # 1. Random Forest
        print("\n[1/3] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        rf_model.fit(X_train_scaled, y_train)
        self.models['Random Forest'] = rf_model
        print("[OK] Random Forest trained!")
        
        # 2. XGBoost
        if XGBOOST_AVAILABLE:
            print("\n[2/3] Training XGBoost...")
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                verbosity=0
            )
            xgb_model.fit(X_train_scaled, y_train)
            self.models['XGBoost'] = xgb_model
            print("[OK] XGBoost trained!")
        else:
            print("\n[2/3] XGBoost skipped (not available)")
        
        # 3. LightGBM
        if LIGHTGBM_AVAILABLE:
            print("\n[3/3] Training LightGBM...")
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            lgb_model.fit(X_train_scaled, y_train)
            self.models['LightGBM'] = lgb_model
            print("[OK] LightGBM trained!")
        else:
            print("\n[3/3] LightGBM skipped (not available)")
        
        print(f"\n[OK] {len(self.models)} model(s) trained successfully!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models
        """
        print("\n" + "="*70)
        print("EVALUATION - MODEL COMPARISON")
        print("="*70)
        
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Evaluating {model_name}")
            print(f"{'='*70}")
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            # Detection rates
            normal_total = (y_test == 0).sum()
            dos_total = (y_test == 1).sum()
            normal_correct = (y_pred[y_test == 0] == 0).sum() / normal_total if normal_total > 0 else 0
            dos_correct = (y_pred[y_test == 1] == 1).sum() / dos_total if dos_total > 0 else 0
            
            print(f"\nPerformance Metrics:")
            print(f"  Accuracy:   {accuracy*100:6.2f}%")
            print(f"  Precision:  {precision*100:6.2f}%")
            print(f"  Recall:     {recall*100:6.2f}%")
            print(f"  F1-Score:   {f1*100:6.2f}%")
            
            print(f"\nDetection Rates:")
            print(f"  Normal Traffic:  {normal_correct*100:6.2f}%")
            print(f"  DoS Detection:    {dos_correct*100:6.2f}%")
            
            print(f"\nConfusion Matrix:")
            print(f"  True Positives (Normal):   {tp:6,}")
            print(f"  True Negatives (DoS):      {tn:6,}")
            print(f"  False Positives:           {fp:6,}")
            print(f"  False Negatives:           {fn:6,}")
            
            # ROC curve and AUC
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
            else:
                fpr, tpr, roc_auc = None, None, None
            
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'normal_rate': normal_correct,
                'dos_rate': dos_correct,
                'cm': cm,
                'y_pred': y_pred,
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': roc_auc,
                'model': model
            }
        
        return results
    
    def visualize_comparison(self, results, X_test, y_test):
        """
        Create comparison visualization
        """
        print("\n" + "="*70)
        print("GENERATING COMPARISON VISUALIZATION")
        print("="*70)
        
        n_models = len(results)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, n_models, hspace=0.3, wspace=0.3)
        
        model_names = list(results.keys())
        
        for idx, model_name in enumerate(model_names):
            result = results[model_name]
            cm = result['cm']
            
            # Confusion Matrix
            ax1 = fig.add_subplot(gs[0, idx])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                        xticklabels=['DoS', 'Normal'], yticklabels=['DoS', 'Normal'])
            ax1.set_title(f'{model_name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
            
            # Performance Metrics
            ax2 = fig.add_subplot(gs[1, idx])
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [result['accuracy'], result['precision'], result['recall'], result['f1']]
            colors = ['#27ae60' if v > 0.9 else '#f39c12' if v > 0.8 else '#e74c3c' for v in values]
            bars = ax2.barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
            ax2.set_xlim(0, 1.05)
            ax2.set_xlabel('Score')
            ax2.set_title(f'{model_name}\nPerformance', fontsize=12, fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            for bar, val in zip(bars, values):
                ax2.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{val*100:.1f}%', va='center', fontweight='bold')
            
            # Detection Rates
            ax3 = fig.add_subplot(gs[2, idx])
            rates = [result['normal_rate'] * 100, result['dos_rate'] * 100]
            labels = ['Normal\nTraffic', 'DoS\nDetection']
            colors_pie = ['#3498db', '#e74c3c']
            wedges, texts, autotexts = ax3.pie(rates, labels=labels, autopct='%1.1f%%',
                                                 colors=colors_pie, startangle=90,
                                                 textprops={'fontweight': 'bold'})
            ax3.set_title(f'{model_name}\nDetection Rates', fontsize=12, fontweight='bold')
        
        plt.savefig('Supervised_DoS_Detection_Comparison.png', dpi=150, bbox_inches='tight')
        print("[OK] Comparison visualization saved: Supervised_DoS_Detection_Comparison.png")
        
        try:
            plt.show()
        except:
            pass
        
        plt.close()
        
        # Create summary comparison
        self.create_summary_table(results)
        
        # Create additional visualizations
        self.create_roc_curves(results, y_test)
        self.create_feature_importance(results)
    
    def create_summary_table(self, results):
        """
        Create summary comparison table
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'DoS Detection':<15}")
        print("-" * 85)
        
        for model_name, result in results.items():
            print(f"{model_name:<20} {result['accuracy']*100:>10.2f}%  {result['precision']*100:>10.2f}%  "
                  f"{result['recall']*100:>10.2f}%  {result['f1']*100:>10.2f}%  {result['dos_rate']*100:>13.2f}%")
        
        # Find best model
        best_f1 = max(results.items(), key=lambda x: x[1]['f1'])
        best_dos = max(results.items(), key=lambda x: x[1]['dos_rate'])
        
        print("\n" + "="*70)
        print("BEST MODELS")
        print("="*70)
        print(f"Best F1-Score: {best_f1[0]} ({best_f1[1]['f1']*100:.2f}%)")
        print(f"Best DoS Detection: {best_dos[0]} ({best_dos[1]['dos_rate']*100:.2f}%)")
    
    def create_roc_curves(self, results, y_test):
        """
        Create ROC curves for all models
        """
        print("\n[INFO] Creating ROC curves...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, result in results.items():
            if result['fpr'] is not None and result['tpr'] is not None:
                ax.plot(result['fpr'], result['tpr'], 
                       label=f"{model_name} (AUC = {result['roc_auc']:.4f})",
                       linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.savefig('ROC_Curves_Comparison.png', dpi=150, bbox_inches='tight')
        print("[OK] ROC curves saved: ROC_Curves_Comparison.png")
        plt.close()
    
    def create_feature_importance(self, results):
        """
        Create feature importance plots
        """
        print("\n[INFO] Creating feature importance plots...")
        
        # Use XGBoost if available, otherwise Random Forest
        model_to_use = None
        model_name = None
        
        if 'XGBoost' in results:
            model_to_use = results['XGBoost']['model']
            model_name = 'XGBoost'
        elif 'LightGBM' in results:
            model_to_use = results['LightGBM']['model']
            model_name = 'LightGBM'
        elif 'Random Forest' in results:
            model_to_use = results['Random Forest']['model']
            model_name = 'Random Forest'
        
        if model_to_use is None:
            print("[WARNING] No model available for feature importance")
            return
        
        # Get feature importance
        if hasattr(model_to_use, 'feature_importances_'):
            importances = model_to_use.feature_importances_
        elif hasattr(model_to_use, 'get_feature_importance'):
            importances = model_to_use.get_feature_importance()
        else:
            print("[WARNING] Model does not support feature importance")
            return
        
        # Create DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Plot top 20 features
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = feature_imp_df.head(20)
        
        ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue', edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=9)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top 20 Feature Importance - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('Feature_Importance_Top20.png', dpi=150, bbox_inches='tight')
        print("[OK] Feature importance saved: Feature_Importance_Top20.png")
        plt.close()
        
        # Save full feature importance to CSV
        feature_imp_df.to_csv('Feature_Importance_Full.csv', index=False)
        print("[OK] Full feature importance saved: Feature_Importance_Full.csv")


def main():
    """
    Main execution - Supervised Learning
    """
    print("\n" + "="*70)
    print("DoS DETECTION - SUPERVISED LEARNING APPROACH")
    print("Methods: XGBoost, Random Forest, LightGBM")
    print("="*70)
    
    # Find CSV files in dataset folder
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        print(f"\n[ERROR] Dataset folder '{dataset_dir}' not found")
        return
    
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv') and 'ISCX' in f]
    
    if not csv_files:
        print(f"\n[ERROR] No CSV files found in '{dataset_dir}' folder")
        return
    
    # Find files
    normal_file = None
    attack_files = []
    
    for f in csv_files:
        if 'Monday' in f:
            normal_file = os.path.join(dataset_dir, f)
        elif 'Wednesday' in f or 'DDos' in f or 'DDoS' in f:
            attack_files.append(os.path.join(dataset_dir, f))
    
    if not attack_files:
        print("\n[ERROR] Attack files not found")
        return
    
    print(f"\nUsing files:")
    if normal_file:
        print(f"  Normal: {normal_file}")
    print(f"  Attack files:")
    for af in attack_files:
        print(f"    - {af}")
    
    try:
        # Initialize detector
        detector = SupervisedDoSDetector()
        
        # Load and prepare data
        X, y = detector.load_and_prepare(
            normal_file=normal_file if normal_file else None,
            attack_files=attack_files,
            sample_size=30000
        )
        
        # Split train/test (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTrain/Test Split:")
        print(f"  Training: {len(X_train):,} samples")
        print(f"  Testing:  {len(X_test):,} samples")
        
        # Train models
        detector.train_models(X_train, y_train)
        
        # Evaluate
        results = detector.evaluate_models(X_test, y_test)
        
        # Visualize
        detector.visualize_comparison(results, X_test, y_test)
        
        print("\n" + "="*70)
        print("[OK] SUPERVISED LEARNING COMPLETED!")
        print("="*70)
        print("\nGenerated files:")
        print("  [GRAPH] Supervised_DoS_Detection_Comparison.png")
        print("\nCompare with Isolation Forest results!")
        
    except Exception as e:
        print(f"\n[ERROR] Error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

