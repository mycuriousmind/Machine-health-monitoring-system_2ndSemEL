import model_utils

cnn_model, rf_model, mu, sd = model_utils.load_all_models()
rf_prob = model_utils.predict_tabular(rf_model, temperature=35.0, rpm=3000, torque=40.0, tool_wear=100.0)
print("Healthy RF Prob:", rf_prob)
