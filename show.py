import joblib
import xgboost as xgb
import matplotlib.pyplot as plt

model = joblib.load("cost_model.pkl")
xgb.plot_tree(model, tree_idx=0, rankdir='LR')

plt.savefig("tree_0.png", dpi=500)
print("Tree saved as tree_0.png")