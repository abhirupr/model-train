import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import average_precision_score
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval
from hyperopt.early_stop import no_progress_loss

class HyperOptXGBoostClassifier:
  def __init__(self, X_train, X_test, y_train, y_test):
    self.X_train = X_train
    self.X_test = X_test
    self.y_test = y_test
    self.y_train = y_train
    self.weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    self.scale = self.weights[1]/self.weights[0]
    self.search_space = {'n_estimators' : hp.quniform('n_estimators', 100, 1000, 100)
,'max_depth' : hp.quniform('max_depth', 1, 30, 1)                                 # depth of trees (preference is for shallow trees or even stumps (max_depth=1))
,'learning_rate' : hp.loguniform('learning_rate',  np.log(0.001), np.log(1))      # learning rate for XGBoost
,'gamma': hp.quniform('gamma', 0.0, 1.0, 0.001)                                   # minimum loss reduction required to make a further partition on a leaf node
,'min_child_weight' : hp.quniform('min_child_weight', 1, 20, 1)                   # minimum number of instances per node
,'subsample' : hp.loguniform('subsample', np.log(0.1), np.log(1.0))               # random selection of rows for training,
,'colsample_bytree' : hp.loguniform('colsample_bytree', np.log(0.1), np.log(1.0)) # proportion of columns to use per tree
,'colsample_bylevel': hp.loguniform('colsample_bylevel', np.log(0.1), np.log(1.0))# proportion of columns to use per level
,'colsample_bynode' : hp.loguniform('colsample_bynode', np.log(0.1), np.log(1.0)) # proportion of columns to use per node
,'scale_pos_weight' : hp.loguniform('scale_pos_weight', np.log(self.scale), np.log(self.scale * 10))   # weight to assign positive label to manage imbalance
}
    self.best_params = None
      
    
  def evaluate_model(self, hyperopt_params):

    # accesss replicated input data
    X_train_input = self.X_train.values
    y_train_input = self.y_train.values
    X_test_input = self.X_test.values
    y_test_input = self.y_test.values

    # configure model parameters
    params = hyperopt_params
    
    if 'n_estimators' in params: params['n_estimators']=int(params['n_estimators']) 
    if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
    if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
    if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
    # all other hyperparameters are taken as given by hyperopt

    params['tree_method'] = 'hist'
    params['grow_policy'] = 'lossguide' #for leaf based tree deveopment
    params['eval_metric'] = 'logloss'
    params['booster'] = 'gbtree'

    # instantiate model with parameters
    model = XGBClassifier(**params)

    # train
    model.fit(X_train_input, y_train_input)

    # predict
    y_prob = model.predict_proba(X_test_input)

    # score
    model_ap_test = average_precision_score(y_test_input, y_prob[:,1])

    # invert metric for hyperopt
    loss = -1 * model_ap_test  
    
    # return results
    return {'loss': loss, 'status': STATUS_OK}
  
  def optimize(self, max_evals=4000, seed=2000, verbose=True, parallelism=4, early_stopping_round=10):
    self.best_params = fmin(
      fn = self.evaluate_model,
      space = self.search_space,
      algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
      max_evals=max_evals,
      trials=SparkTrials(parallelism=parallelism),
      early_stop_fn = self.no_progress_loss_custom(early_stopping_round),
      rstate=np.random.default_rng(seed), 
      verbose=verbose
    )
    self.best_params['n_estimators'] = int(self.best_params['n_estimators'])
    self.best_params['max_depth'] = int(self.best_params['max_depth'])
    self.best_params['min_child_weight'] = int(self.best_params['min_child_weight'])
    self.best_params['scale_pos_weight'] = int(self.best_params['scale_pos_weight'])

  def no_progress_loss_custom(iteration_stop_count=20, percent_increase=0.0):
    def stop_fn(trials, best_loss=None, iteration_no_progress=0):
      # print(trials.trials[-1])
      if trials.trials[-1]["result"]["status"] == "new":
        return (False, [None, 0])
      new_loss = trials.trials[-1]["result"]["loss"]
      if best_loss is None:
          return False, [new_loss, iteration_no_progress + 1]
      best_loss_threshold = best_loss - abs(best_loss * (percent_increase / 100.0))
      if new_loss < best_loss_threshold:
          best_loss = new_loss
          iteration_no_progress = 0
      else:
          iteration_no_progress += 1

      return (
          iteration_no_progress >= iteration_stop_count,
          [best_loss, iteration_no_progress],
      )

    return stop_fn
  
  def train_best_model(self):
    if self.best_params is None:
      raise ValueError("Hyperparameters not optimized yet. Call `optimize` first.")
    else:
      self.model = XGBClassifier(
            **self.best_params,
            tree_method='hist',
            eval_metric='logloss',
            grow_policy='lossguide',
            booster= 'gbtree'
        )
    self.model.fit(self.X_train, self.y_train)

    def predict(self, df):
      if not hasattr(self, 'model'):
        raise ValueError("Model not trained yet. Call `train_best_model` first.")
      return self.model.predict_proba(df)