xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.savefig(plot_path + 'model_decision_tree.png')

plt.show()


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [50, 50]
plt.savefig(plot_path + 'feature_importance.png')

plt.show()
