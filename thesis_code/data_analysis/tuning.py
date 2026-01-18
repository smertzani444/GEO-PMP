# ======================================================
# 4. Hyperparameter optimization objective
# ======================================================

def rf_objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 5, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical(
            "max_features", ["sqrt", "log2"]
        ),
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    }

    model = RandomForestClassifier(**params)

    class_weights = calculate_class_weights(y_train)

    model = train_model(
        model,
        X_train,
        y_train,
        class_weights=class_weights
    )

    y_val_proba = model.predict_proba(X_val)[:, 1]

    return average_precision_score(y_val, y_val_proba)