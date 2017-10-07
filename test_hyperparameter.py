import utils.datasets as ds
import models.gan
import json
import numpy as np
import tensorflow as tf
import os

params = [10, 50, 100, 500, 1000, 2000, 5000]
hyperparam_name = 'num_epochs'

hyperparameters = dict(
    num_features=12, num_epochs=1000, normalize='rescaling',
    debug=True, latent_vector_size=100,
    batch_size=1000, ns_param=.5, adpt_l=0,
    res_depth=1, dr_param=1, batch_param=1e-2,
    display_step=10, d_learning_rate=1e-3,
    reg_param=1e-3, g_learning_rate=1e-4
)

loc_str = 'results/{}'.format(hyperparam_name)
loc_str += '/trial_{}/{}.json'


def get_summary(data):
    out = {}

    mats = np.array([entry['confusion_matrix'] for entry in data])

    out['avg_acc'] = np.mean([entry['accuracy'] for entry in data])
    out['std_acc'] = np.std([entry['accuracy'] for entry in data])
    out['avg_mat'] = np.mean(mats, axis=0).tolist()
    out['std_mat'] = np.std(mats, axis=0).tolist()

    return out


loc = 'results/{}'.format(hyperparam_name)
if not os.path.exists(loc):
    os.mkdir(loc)

exploits = ['freak', 'nginx_keyleak', 'nginx_rootdir', 'caleb']

for param in params:
    hyperparameters[hyperparam_name] = param

    summaries = {'hyperparameters': hyperparameters}
    raw_data = []

    dirs = [el for el in os.listdir('results/{}'.format(hyperparam_name))
            if 'trial_' in el]
    trials = [int(el.split('_')[1]) for el in dirs]
    trials.insert(0, -1)
    trial = np.max(trials) + 1
    os.mkdir('results/{}/trial_{}'.format(hyperparam_name, trial))

    with tf.Graph().as_default():
        model = models.gan.GAN(**hyperparameters)

        for exploit in exploits:
            data = []

            for i in range(5):

                trX, trY = ds.load_data(
                    (
                        './data/three-step/{}/subset_{}/train_set.csv'
                    ).format(exploit, i)
                )

                model.train(trX, trY)

                for i in range(5):
                    teX, teY = ds.load_data(
                        (
                            './data/three-step/{}/subset_{}/test_set.csv'
                        ).format(exploit, i)
                    )

                    d = model.test(teX, teY)
                    data.append(d)
                    raw_data.append(d)

            summaries[exploit] = get_summary(data)

            with open(loc_str.format(trial, exploit), 'w') as f:
                json.dump(data, f, indent=2)

        summaries['net'] = get_summary(raw_data)

    with open(loc_str.format(trial, 'summary'), 'w') as f:
        json.dump(summaries, f, indent=2)
