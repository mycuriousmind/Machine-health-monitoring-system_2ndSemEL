import model_utils
import numpy as np

cnn_model, rf_model, mu_vib, sd_vib = model_utils.load_all_models()

# Test CNN
t = np.linspace(0, 100 / 12000, 100)
f0 = 30.0 
bpfi = f0 * 5.4
sig = 1.0 * np.sin(2*np.pi*f0*t) + 0.4 * np.sin(2*np.pi*bpfi*t) + 0.2 * np.sin(2*np.pi*2*bpfi*t)
for pos in [20, 60]:
    sig += 2.0 * np.exp(-200*(np.arange(100)-pos)**2/100)
raw_data = np.stack([sig, sig, sig], axis=1).astype(np.float32) + np.random.randn(100, 3).astype(np.float32) * 0.2

cnn_prob = model_utils.predict_vibration(cnn_model, raw_data, mu_vib, sd_vib)
print("CNN Prob:", cnn_prob)

# Test RF
rf_prob = model_utils.predict_tabular(rf_model, temperature=95.0, rpm=8500, torque=80.0, tool_wear=220.0)
print("RF Prob:", rf_prob)

