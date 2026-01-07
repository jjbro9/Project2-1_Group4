import matplotlib.pyplot as plt
import pandas as pd

feature_importances_dict = {
    'behavior_name': 0.0003,
    'algorithm': 0.1257,
    'batch_size': 0.1863,
    'buffer_size': 0.2295,
    'learning_rate': 0.0131,
    'betaepsilon': 0.1032,
    'lambd': 0.0694,
    'num_epoch': 0.0864,
    'learning_rate_schedule': 0.0002,
    'normalize': 0.0000,
    'hidden_units': 0.0168,
    'num_layers': 0.1152,
    'gamma': 0.0124,
    'reward_strength': 0.0127,
    'max_steps': 0.0000,
    'time_horizon': 0.0098,
    'summary_freq': 0.0000,
    'seed': 0.0190,
    'cpu_count': 0.0000
}

fi_series = pd.Series(feature_importances_dict)
top_features = fi_series.sort_values(ascending=False).head(15)

plt.figure(figsize=(10,6))
top_features.plot(kind='barh', color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.show()

