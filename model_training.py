import numpy as np
import os
import joblib

from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics         import r2_score, mean_absolute_error, mean_squared_error


def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
   
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain / Test split: {len(X_train)} / {len(X_test)} customers")
    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train) -> dict:
  
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
        ),
    }

    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"  ✓ {name}")

    return models


def evaluate_models(models: dict, X_train, X_test, y_train, y_test, X_all, y_all) -> list:
    
    results = []
    print("\n" + "=" * 65)
    print("MODEL EVALUATION")
    print("=" * 65)

    for name, model in models.items():
    
        y_pred_test  = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        test_r2   = r2_score(y_test, y_pred_test)
        test_mae  = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_r2  = r2_score(y_train, y_pred_train)

        cv_scores = cross_val_score(model, X_all, y_all, cv=5, scoring='r2', n_jobs=-1)
        cv_mean   = cv_scores.mean()
        cv_std    = cv_scores.std()

        overfit_gap = train_r2 - test_r2

        result = {
            'name':        name,
            'train_r2':    round(train_r2, 4),
            'test_r2':     round(test_r2, 4),
            'mae':         round(test_mae, 2),
            'rmse':        round(test_rmse, 2),
            'cv_mean':     round(cv_mean, 4),
            'cv_std':      round(cv_std, 4),
            'overfit_gap': round(overfit_gap, 4),
        }
        results.append(result)

        overfit_flag = "⚠️ possible overfit" if overfit_gap > 0.05 else "✓ stable"
        print(f"\n  {name}")
        print(f"    Test R²:  {test_r2:.4f}    Train R²: {train_r2:.4f}  {overfit_flag}")
        print(f"    MAE:     ${test_mae:>8,.2f}    RMSE:    ${test_rmse:>8,.2f}")
        print(f"    CV R²:    {cv_mean:.4f} ± {cv_std:.4f}")

  
    results.sort(key=lambda r: r['cv_mean'], reverse=True)

    print(f"\n  🏆 Best model: {results[0]['name']}  (CV R²={results[0]['cv_mean']})")
    return results


def get_feature_importance(model, feature_names: list) -> list:
  
    if not hasattr(model, 'feature_importances_'):
        return []

    importances = model.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nFeature Importances (Random Forest):")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {imp:.4f}  {bar}")

    return [{'feature': f, 'importance': round(float(v), 4)} for f, v in feat_imp]


def get_predictions(model, X_test, y_test) -> dict:
  
    y_pred = model.predict(X_test)
    return {
        'actual':    y_test.values.tolist(),
        'predicted': y_pred.tolist(),
        'residuals': (y_test.values - y_pred).tolist(),
    }


def save_models(models: dict, output_dir: str = 'models') -> None:
  
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        path = os.path.join(output_dir, filename)
        joblib.dump(model, path)
        print(f"  Saved: {path}")


def run_full_modeling(X, y, feature_names: list, models_dir: str = 'models') -> dict:

    print("\n" + "█" * 55)
    print("  MACHINE LEARNING — CLV PREDICTION")
    print("█" * 55)

    X_train, X_test, y_train, y_test = split_data(X, y)
    models  = train_all_models(X_train, y_train)
    results = evaluate_models(models, X_train, X_test, y_train, y_test, X, y)

    best_name  = results[0]['name']
    best_model = models[best_name]

    feat_imp   = get_feature_importance(best_model, feature_names)
    predictions = get_predictions(
        models['Gradient Boosting'],  # Always use GB for the actual-vs-predicted plot
        X_test, y_test
    )

    print("\nSaving models...")
    save_models(models, models_dir)

    return {
        'models':       models,
        'results':      results,
        'best_model':   best_model,
        'best_name':    best_name,
        'feat_imp':     feat_imp,
        'predictions':  predictions,
        'X_test':       X_test,
        'y_test':       y_test,
    }
