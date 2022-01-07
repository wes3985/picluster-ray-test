import numpy as np 
import pandas as pd 
#import tensorflow
from sklearn import linear_model 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV, TimeSeriesSplit 
from sklearn.preprocessing import StandardScaler 
from sklearn.kernel_ridge import KernelRidge 
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import ray 
from ray.cluster_utils import Cluster
import psutil
#statsmodels

# generate some data
length = 10000
df = pd.DataFrame()
x = np.sin(np.arange(length)/60.0)
x2 = [np.random.uniform(-10,10) for x in range(length)]
df['x2'] = x2
df['x'] = df['x2'].cumsum()
df['y'] = df['x'].shift(-1)
df['y'] = df['y'].fillna(0)
#print(df)
#df.to_csv("x.csv")

lb=780
w=130


try:
        
    try:
        OSM = int(ray.utils.get_system_memory() * 0.3 )
    except Exception:
        OSM = 5000000000

    if not ray.is_initialized():
        cluster = Cluster(
            initialize_head=True,
            head_node_args={"num_cpus": psutil.cpu_count(),
                            "object_store_memory":OSM}
        )

        # start ray
        ray.init(address=cluster.address, include_dashboard=False, log_to_driver=False) 



    def workflow(df, d, lb, w):

        # generate some models
        models = {"svr":SVR(), "en":linear_model.ElasticNet(), "rfr":RandomForestRegressor()}

        results = {"d":d, "svr":np.nan, "en":np.nan, "rfr":np.nan}

        # prep data
        data = df.loc[d-lb:d+w, ["x","y"]]

        for x in range(1, w+1):
            data.iloc[-x]["y"] = np.nan     
        #print(data[~(np.isnan(data['y']))])

        # Scale the data 
        scaler = StandardScaler()
        data_s = scaler.fit_transform(data)

        x = data_s[:, :-1]
        y = data_s[:, -1]
        x_train = x[:lb]
        y_train = y[:lb]
        x_test = x[lb+1:]

        try:
            assert(len(x_train)==lb)
            assert(len(x_test)==w)   
        except AssertionError:
            for model in models:
                results[model] = np.nan
            return results


        for model in models:
            models[model].fit(x_train, y_train)
            this_prediction = [scaler.inverse_transform([[0]*len(x_test[0])+[p]]).flatten()[-1] for p in models[model].predict(x_test)]
            this_prediction = np.sum(this_prediction)
            results[model] = this_prediction

        return results

    @ray.remote(num_cpus=1)
    def ray_workflow(d, lb, w):

        df = ray.get(df_id)

        results = workflow(df, d, lb, w)

        return results

    df_id = ray.put(df)

    result_ids = [ray_workflow.remote(d, lb, w) for d in range(lb, len(df))]

    # Loop over pending results and process completed 
    completed=0
    conclusions = []
    while len(result_ids):
        done_id, result_ids = ray.wait(result_ids)
        sp = ray.get(done_id[0])


        conclusions.append(sp)


        completed+=1

        print(completed)


    out = pd.DataFrame(conclusions)
    out.to_csv("out.csv")


except Exception as e:
    raise e

finally:
    if ray.is_initialized():
        ray.shutdown()