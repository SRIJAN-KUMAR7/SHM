y_pred = model.predict(X_test)
print("R² Score on Test Set:", r2_score(y_test, y_pred))
